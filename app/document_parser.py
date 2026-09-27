import csv
import importlib.util
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree

from .document_compression import DocumentCompressor
from .models import DocumentChunk, OCRSpan, ParsedDocument
from .text_chunking import split_text_naturally


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}
HEADING_PATTERN = re.compile(
    r"^(?:#{1,6}\s+.+|\d+(?:\.\d+){0,4}\s+.+|[一二三四五六七八九十]+、.+)$"
)
USEFUL_CHARACTER_PATTERN = re.compile(r"[A-Za-z0-9\u3400-\u9fff]")
SPACE_PATTERN = re.compile(r"\s+")


@dataclass
class OCRSpanData:
    text: str
    bbox: List[List[float]]
    confidence: float


@dataclass
class PageExtraction:
    text: str
    extraction_method: str = "native"
    native_text_chars: int = 0
    ocr_text_chars: int = 0
    ocr_spans: List[OCRSpanData] = field(default_factory=list)


class AutoOCRBackend:
    """Lazy OCR adapter: PaddleOCR first, command-line Tesseract second."""

    def __init__(self, language: str = "ch") -> None:
        self.language = language
        self._engine = None
        self._engine_name = self._detect_engine()

    @staticmethod
    def _detect_engine() -> str:
        if importlib.util.find_spec("paddleocr"):
            return "paddleocr"
        if shutil.which("tesseract"):
            return "tesseract"
        return ""

    @property
    def available(self) -> bool:
        return bool(self._engine_name)

    @property
    def name(self) -> str:
        return self._engine_name or "unavailable"

    def extract(self, image_path: Path) -> List[OCRSpanData]:
        if self._engine_name == "paddleocr":
            return self._extract_paddle(image_path)
        if self._engine_name == "tesseract":
            return self._extract_tesseract(image_path)
        return []

    def _extract_paddle(self, image_path: Path) -> List[OCRSpanData]:
        if self._engine is None:
            from paddleocr import PaddleOCR

            try:
                self._engine = PaddleOCR(
                    use_angle_cls=True, lang=self.language, show_log=False
                )
            except TypeError:
                self._engine = PaddleOCR(use_angle_cls=True, lang=self.language)
        result = self._engine.ocr(str(image_path), cls=True)
        lines = result[0] if result and isinstance(result[0], list) else result
        spans: List[OCRSpanData] = []
        for line in lines or []:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue
            box, recognition = line[0], line[1]
            if not isinstance(recognition, (list, tuple)) or len(recognition) < 2:
                continue
            text = str(recognition[0]).strip()
            if not text:
                continue
            spans.append(
                OCRSpanData(
                    text=text,
                    bbox=[[float(point[0]), float(point[1])] for point in box],
                    confidence=float(recognition[1]),
                )
            )
        return spans

    def _extract_tesseract(self, image_path: Path) -> List[OCRSpanData]:
        language = "chi_sim+eng" if self.language.startswith("ch") else "eng"
        result = subprocess.run(
            ["tesseract", str(image_path), "stdout", "-l", language, "tsv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode:
            raise ValueError(
                "Tesseract OCR failed: {}".format(
                    result.stderr.decode("utf-8", errors="replace").strip()
                )
            )
        rows = csv.DictReader(
            io.StringIO(result.stdout.decode("utf-8", errors="replace")),
            delimiter="\t",
        )
        groups: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
        for row in rows:
            text = (row.get("text") or "").strip()
            try:
                confidence = float(row.get("conf") or -1)
            except ValueError:
                confidence = -1
            if not text or confidence < 0:
                continue
            key = (
                row.get("page_num") or "0",
                row.get("block_num") or "0",
                row.get("par_num") or "0",
                row.get("line_num") or "0",
            )
            groups.setdefault(key, []).append(row)

        spans: List[OCRSpanData] = []
        for words in groups.values():
            left = min(int(word["left"]) for word in words)
            top = min(int(word["top"]) for word in words)
            right = max(int(word["left"]) + int(word["width"]) for word in words)
            bottom = max(int(word["top"]) + int(word["height"]) for word in words)
            confidence = mean(float(word["conf"]) for word in words) / 100.0
            spans.append(
                OCRSpanData(
                    text=" ".join(word["text"].strip() for word in words),
                    bbox=[
                        [float(left), float(top)],
                        [float(right), float(top)],
                        [float(right), float(bottom)],
                        [float(left), float(bottom)],
                    ],
                    confidence=max(0.0, min(1.0, confidence)),
                )
            )
        return spans


class DocumentParsingAgent:
    """Extract, filter, and compress PRDs for downstream module planning."""

    name = "document_parsing"

    def __init__(self, llm: Any, ocr_backend: Optional[Any] = None) -> None:
        self.llm = llm
        self.ocr_backend = ocr_backend or AutoOCRBackend(
            os.getenv("OCR_LANGUAGE", "ch")
        )
        self.min_native_chars = max(
            1, int(os.getenv("OCR_MIN_NATIVE_CHARS", "80"))
        )
        self.ocr_dpi = max(120, int(os.getenv("OCR_DPI", "220")))
        self.compressor = DocumentCompressor(llm)

    def run(self, payload: Dict[str, Any], context: Dict[str, Any]) -> ParsedDocument:
        if payload.get("document"):
            document = ParsedDocument.model_validate(payload["document"])
        else:
            document = self.extract(payload, context)
        return self.compressor.compress(
            document,
            force_demo=bool(context.get("force_demo")),
        )

    def extract(
        self,
        payload: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ParsedDocument:
        filename = str(payload["filename"])
        data = payload["data"]
        extension = Path(filename).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported document type: {}".format(extension or "unknown"))

        if extension == ".pdf":
            pages, warnings, extraction_summary = self._extract_pdf(data)
        elif extension == ".docx":
            raw_pages, warnings = self._extract_docx(data)
            pages = self._native_pages(raw_pages)
            extraction_summary = self._summary(pages, "docx_xml")
        else:
            raw_pages, warnings = self._extract_text(data)
            pages = self._native_pages(raw_pages)
            extraction_summary = self._summary(pages, "text_decoder")

        chunks = self._chunk_pages(pages)
        normalized = "\n\n".join(
            "[第{}页][{}]\n{}".format(
                chunk.page or "?", chunk.section or "正文", chunk.content
            )
            for chunk in chunks
        )
        document_type = self._detect_document_type(filename, normalized)
        if not chunks:
            warnings.append("文档没有提取到可用文本，请检查 OCR 依赖和文档内容。")
        return ParsedDocument(
            filename=filename,
            document_type=document_type,
            normalized_text=normalized,
            raw_text=normalized,
            chunks=chunks,
            warnings=warnings,
            extraction_summary=extraction_summary,
        )

    def _extract_pdf(
        self, data: bytes
    ) -> Tuple[List[PageExtraction], List[str], Dict[str, Any]]:
        temporary = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        path = Path(temporary.name)
        warnings: List[str] = []
        try:
            temporary.write(data)
            temporary.close()
            native_pages = self._extract_native_pdf(path)
            pages = self._native_pages(native_pages)
            low_text_pages = [
                index
                for index, page in enumerate(pages, 1)
                if self._needs_ocr(page.text)
            ]
            if not low_text_pages:
                return pages, warnings, self._summary(pages, "pdftotext")

            if not self.ocr_backend.available:
                warnings.append(
                    "第 {} 页文本过少，未检测到 PaddleOCR 或 Tesseract，已保留原生提取结果。".format(
                        "、".join(str(page) for page in low_text_pages)
                    )
                )
                summary = self._summary(pages, "pdftotext")
                summary["ocr_candidate_pages"] = low_text_pages
                summary["ocr_backend"] = "unavailable"
                return pages, warnings, summary

            with tempfile.TemporaryDirectory(prefix="caseforge-ocr-") as image_dir:
                try:
                    images = self._render_pdf_pages(
                        path, low_text_pages, Path(image_dir)
                    )
                except (FileNotFoundError, ValueError) as exc:
                    warnings.append("OCR 页面渲染失败：{}".format(exc))
                    summary = self._summary(pages, "pdftotext")
                    summary["ocr_candidate_pages"] = low_text_pages
                    summary["ocr_backend"] = self.ocr_backend.name
                    return pages, warnings, summary

                for page_number in low_text_pages:
                    image_path = images.get(page_number)
                    if image_path is None:
                        warnings.append("第 {} 页未生成 OCR 图像。".format(page_number))
                        continue
                    try:
                        spans = self.ocr_backend.extract(image_path)
                    except Exception as exc:
                        warnings.append(
                            "第 {} 页 OCR 失败：{}".format(page_number, exc)
                        )
                        continue
                    ocr_text = "\n".join(span.text for span in spans if span.text)
                    if not ocr_text.strip():
                        warnings.append(
                            "第 {} 页 OCR 未识别到有效文字。".format(page_number)
                        )
                        continue
                    native_text = pages[page_number - 1].text
                    merged_text = self._merge_native_and_ocr(native_text, ocr_text)
                    method = "ocr" if not native_text.strip() else "hybrid"
                    pages[page_number - 1] = PageExtraction(
                        text=merged_text,
                        extraction_method=method,
                        native_text_chars=self._useful_char_count(native_text),
                        ocr_text_chars=self._useful_char_count(ocr_text),
                        ocr_spans=spans,
                    )

            summary = self._summary(pages, "pdftotext+ocr")
            summary["ocr_backend"] = self.ocr_backend.name
            summary["render_dpi"] = self.ocr_dpi
            return pages, warnings, summary
        finally:
            temporary.close()
            if path.exists():
                path.unlink()

    @staticmethod
    def _extract_native_pdf(path: Path) -> List[str]:
        try:
            result = subprocess.run(
                ["pdftotext", "-layout", str(path), "-"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ValueError("pdftotext is required to parse PDF files") from exc
        if result.returncode:
            raise ValueError(
                "PDF parsing failed: {}".format(
                    result.stderr.decode("utf-8", errors="replace").strip()
                )
            )
        text = result.stdout.decode("utf-8", errors="replace")
        pages = text.split("\f")
        if len(pages) > 1 and not pages[-1].strip():
            pages.pop()
        return [page.strip() for page in pages] or [""]

    def _render_pdf_pages(
        self, pdf_path: Path, page_numbers: List[int], output_dir: Path
    ) -> Dict[int, Path]:
        if importlib.util.find_spec("fitz"):
            import fitz

            images: Dict[int, Path] = {}
            document = fitz.open(str(pdf_path))
            try:
                scale = self.ocr_dpi / 72.0
                matrix = fitz.Matrix(scale, scale)
                for page_number in page_numbers:
                    if page_number < 1 or page_number > len(document):
                        continue
                    output = output_dir / "page-{}.png".format(page_number)
                    pixmap = document[page_number - 1].get_pixmap(
                        matrix=matrix, alpha=False
                    )
                    pixmap.save(str(output))
                    images[page_number] = output
            finally:
                document.close()
            return images

        if not shutil.which("pdftoppm"):
            raise FileNotFoundError(
                "PyMuPDF or pdftoppm is required to render OCR pages"
            )
        images = {}
        for page_number in page_numbers:
            prefix = output_dir / "page-{}".format(page_number)
            result = subprocess.run(
                [
                    "pdftoppm",
                    "-f",
                    str(page_number),
                    "-l",
                    str(page_number),
                    "-r",
                    str(self.ocr_dpi),
                    "-singlefile",
                    "-png",
                    str(pdf_path),
                    str(prefix),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode:
                raise ValueError(
                    result.stderr.decode("utf-8", errors="replace").strip()
                )
            output = Path(str(prefix) + ".png")
            if output.exists():
                images[page_number] = output
        return images

    @staticmethod
    def _extract_docx(data: bytes) -> Tuple[List[str], List[str]]:
        temporary = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
        path = Path(temporary.name)
        try:
            temporary.write(data)
            temporary.close()
            with zipfile.ZipFile(str(path)) as archive:
                xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(xml)
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            paragraphs = []
            for paragraph in root.iter(namespace + "p"):
                value = "".join(
                    node.text or "" for node in paragraph.iter(namespace + "t")
                ).strip()
                if value:
                    paragraphs.append(value)
            return ["\n".join(paragraphs)], [
                "DOCX 无稳定页码信息，证据页码统一标记为第 1 页。"
            ]
        except (KeyError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            raise ValueError("Invalid DOCX document") from exc
        finally:
            temporary.close()
            if path.exists():
                path.unlink()

    @staticmethod
    def _extract_text(data: bytes) -> Tuple[List[str], List[str]]:
        for encoding in ["utf-8-sig", "utf-8", "gb18030"]:
            try:
                return [data.decode(encoding)], []
            except UnicodeDecodeError:
                continue
        return [data.decode("utf-8", errors="replace")], [
            "文本编码无法准确识别，无法解码的字符已被替换。"
        ]

    @staticmethod
    def _native_pages(pages: List[str]) -> List[PageExtraction]:
        return [
            PageExtraction(
                text=page,
                extraction_method="native",
                native_text_chars=DocumentParsingAgent._useful_char_count(page),
            )
            for page in pages
        ]

    def _needs_ocr(self, text: str) -> bool:
        compact = SPACE_PATTERN.sub("", text)
        if not compact:
            return True
        useful = self._useful_char_count(text)
        return useful < self.min_native_chars or useful / len(compact) < 0.35

    @staticmethod
    def _useful_char_count(text: str) -> int:
        return len(USEFUL_CHARACTER_PATTERN.findall(text or ""))

    @staticmethod
    def _merge_native_and_ocr(native_text: str, ocr_text: str) -> str:
        native_lines = [line.strip() for line in native_text.splitlines() if line.strip()]
        merged = list(native_lines)
        normalized_native = {
            SPACE_PATTERN.sub("", line).lower() for line in native_lines
        }
        for line in ocr_text.splitlines():
            clean = line.strip()
            normalized = SPACE_PATTERN.sub("", clean).lower()
            if not normalized:
                continue
            if any(
                normalized == existing
                or (len(normalized) >= 8 and normalized in existing)
                or (len(existing) >= 8 and existing in normalized)
                for existing in normalized_native
            ):
                continue
            merged.append(clean)
            normalized_native.add(normalized)
        return "\n".join(merged)

    @staticmethod
    def _summary(
        pages: List[PageExtraction], parser: str
    ) -> Dict[str, Any]:
        methods: Dict[str, int] = {}
        for page in pages:
            methods[page.extraction_method] = methods.get(page.extraction_method, 0) + 1
        ocr_pages = [
            index
            for index, page in enumerate(pages, 1)
            if page.extraction_method in {"ocr", "hybrid"}
        ]
        confidences = [
            span.confidence
            for page in pages
            for span in page.ocr_spans
            if span.confidence >= 0
        ]
        return {
            "parser": parser,
            "page_count": len(pages),
            "page_methods": methods,
            "ocr_pages": ocr_pages,
            "ocr_backend": (
                "none" if not ocr_pages else "automatic"
            ),
            "average_ocr_confidence": (
                round(mean(confidences), 4) if confidences else None
            ),
        }

    @staticmethod
    def _detect_document_type(filename: str, content: str) -> str:
        sample = (filename + "\n" + content[:5000]).lower()
        rules = [
            ("test_plan", ["测试计划", "测试方案", "test plan"]),
            ("api_spec", ["openapi", "swagger", "接口文档", "api spec"]),
            ("technical_design", ["技术方案", "架构设计", "technical design"]),
            ("prd", ["prd", "产品需求", "需求文档"]),
        ]
        for document_type, keywords in rules:
            if any(keyword in sample for keyword in keywords):
                return document_type
        return "general_document"

    @staticmethod
    def _chunk_pages(
        pages: List[PageExtraction], max_chars: int = 1600
    ) -> List[DocumentChunk]:
        chunks: List[DocumentChunk] = []
        chunk_index = 1
        for page_number, page in enumerate(pages, 1):
            section = "正文"
            buffer: List[str] = []

            def flush() -> None:
                nonlocal chunk_index
                content = "\n".join(buffer).strip()
                if not content:
                    return
                normalized_content = SPACE_PATTERN.sub("", content).lower()
                relevant_spans = [
                    span
                    for span in page.ocr_spans
                    if SPACE_PATTERN.sub("", span.text).lower() in normalized_content
                ]
                confidence = (
                    round(mean(span.confidence for span in relevant_spans), 4)
                    if relevant_spans
                    else None
                )
                chunks.append(
                    DocumentChunk(
                        id="DOC-P{}-C{}".format(page_number, chunk_index),
                        page=page_number,
                        section=section,
                        content=content,
                        extraction_method=page.extraction_method,
                        ocr_confidence=confidence,
                        ocr_spans=[
                            OCRSpan(
                                text=span.text,
                                bbox=span.bbox,
                                confidence=round(span.confidence, 4),
                            )
                            for span in relevant_spans
                        ],
                    )
                )
                chunk_index += 1
                buffer.clear()

            for raw_line in page.text.splitlines():
                line = SPACE_PATTERN.sub(" ", raw_line).strip()
                if not line:
                    continue
                if HEADING_PATTERN.match(line) or (
                    len(line) <= 36 and line.endswith("：")
                ):
                    flush()
                    section = line.lstrip("#").strip()
                    continue
                candidate = "\n".join(buffer + [line])
                segments = split_text_naturally(candidate, max_chars)
                if len(segments) == 1:
                    buffer[:] = segments
                    continue
                for segment in segments[:-1]:
                    buffer[:] = [segment]
                    flush()
                buffer[:] = [segments[-1]]
            flush()
        return chunks
