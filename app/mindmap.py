from collections import defaultdict
from typing import Dict, Iterator, List

from .models import MindMapDocument, MindMapNode, ProjectState, TestCase, TestModule


class MindMapConversionTool:
    """Convert structured modules and cases into a presentation-neutral mind map."""

    name = "mindmap_conversion"

    def schema(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": "Convert a module tree or generated test cases into a mind-map tree.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "project_id": {"type": "string"},
                    "view": {"type": "string", "enum": ["modules", "cases"]},
                },
                "required": ["project_id", "view"],
            },
            "side_effect": "none",
        }

    def convert(self, project: ProjectState, view: str = "cases") -> MindMapDocument:
        if view not in {"modules", "cases"}:
            raise ValueError("Mind-map view must be 'modules' or 'cases'.")
        if not project.module_tree:
            raise ValueError("A module tree is required before mind-map conversion.")

        cases_by_module: Dict[str, List[TestCase]] = defaultdict(list)
        for case in project.cases:
            cases_by_module[case.module_id].append(case)

        known_module_ids = {
            module.id
            for root in project.module_tree.modules
            for module in self._walk_modules(root)
        }
        module_nodes = [
            self._module_node(module, view, cases_by_module)
            for module in project.module_tree.modules
        ]
        orphan_cases = [
            case for case in project.cases if case.module_id not in known_module_ids
        ]
        if view == "cases" and orphan_cases:
            module_nodes.append(
                MindMapNode(
                    id="module-unassigned",
                    label="Unassigned cases",
                    node_type="module",
                    metadata={"case_count": len(orphan_cases)},
                    children=[self._case_node(case) for case in orphan_cases],
                )
            )

        return MindMapDocument(
            project_id=project.id,
            view=view,
            root=MindMapNode(
                id=f"project-{project.id}",
                label=project.title,
                node_type="project",
                metadata={"phase": project.phase},
                children=module_nodes,
            ),
            stats={
                "modules": len(known_module_ids),
                "cases": len(project.cases) if view == "cases" else 0,
                "orphan_cases": len(orphan_cases) if view == "cases" else 0,
            },
        )

    def _module_node(
        self,
        module: TestModule,
        view: str,
        cases_by_module: Dict[str, List[TestCase]],
    ) -> MindMapNode:
        children = [
            self._module_node(child, view, cases_by_module)
            for child in module.children
        ]
        direct_cases = cases_by_module.get(module.id, [])
        if view == "cases":
            children.extend(self._case_node(case) for case in direct_cases)

        return MindMapNode(
            id=f"module-{module.id}",
            label=module.name,
            node_type="module",
            metadata={
                "module_id": module.id,
                "objective": module.objective,
                "requirement_ids": module.requirement_ids,
                "risks": module.risks,
                "case_types": module.case_types,
                "direct_case_count": len(direct_cases),
            },
            children=children,
        )

    def _case_node(self, case: TestCase) -> MindMapNode:
        children: List[MindMapNode] = []
        if case.preconditions:
            children.append(
                MindMapNode(
                    id=f"{case.id}-preconditions",
                    label="Preconditions",
                    node_type="group",
                    children=[
                        MindMapNode(
                            id=f"{case.id}-precondition-{index}",
                            label=value,
                            node_type="precondition",
                        )
                        for index, value in enumerate(case.preconditions, start=1)
                    ],
                )
            )

        if case.steps:
            step_nodes = []
            for index, step in enumerate(case.steps, start=1):
                step_nodes.append(
                    MindMapNode(
                        id=f"{case.id}-step-{index}",
                        label=f"{index}. {step.action}",
                        node_type="step",
                        children=[
                            MindMapNode(
                                id=f"{case.id}-expected-{index}",
                                label=step.expected,
                                node_type="expected",
                            )
                        ],
                    )
                )
            children.append(
                MindMapNode(
                    id=f"{case.id}-steps",
                    label="Steps and expected results",
                    node_type="group",
                    children=step_nodes,
                )
            )

        if case.test_data:
            children.append(
                MindMapNode(
                    id=f"{case.id}-test-data",
                    label="Test data",
                    node_type="group",
                    children=[
                        MindMapNode(
                            id=f"{case.id}-data-{index}",
                            label=f"{key}: {value}",
                            node_type="test_data",
                        )
                        for index, (key, value) in enumerate(
                            case.test_data.items(), start=1
                        )
                    ],
                )
            )

        return MindMapNode(
            id=f"case-{case.id}",
            label=case.title,
            node_type="case",
            metadata={
                "case_id": case.id,
                "priority": case.priority,
                "case_type": case.case_type,
                "requirement_ids": case.requirement_ids,
                "risk_tags": case.risk_tags,
            },
            children=children,
        )

    def _walk_modules(self, module: TestModule) -> Iterator[TestModule]:
        yield module
        for child in module.children:
            yield from self._walk_modules(child)
