import ast
from pathlib import Path
from types import SimpleNamespace


def test_close_destroys_both_owned_groups_and_only_owned_comm_ids():
    path = Path(__file__).parents[3] / "afd_plugin/connectors/gpu/p2p.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "P2pNcclAFDConnector"
    )
    tree.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "close"
    ]
    events = []
    registry = {1: "a2f", 2: "f2a", 3: "another_connector"}
    namespace = {
        "_AFD_COMMUNICATORS": registry,
        "torch": SimpleNamespace(
            distributed=SimpleNamespace(
                destroy_process_group=lambda group: events.append(group)
            )
        ),
    }
    exec(compile(tree, str(path), "exec"), namespace)
    connector = SimpleNamespace(
        a2e_comm_id=1,
        e2a_comm_id=2,
        a2e_pynccl=SimpleNamespace(destroy=lambda: events.append("a2f")),
        e2a_pynccl=SimpleNamespace(destroy=lambda: events.append("f2a")),
        p2p_pg="control_group",
        afd_pg="afd_world",
        a2e_group="store",
        e2a_group="store",
        _initialized=True,
        dp_metadata_list={0: 1},
        tensor_metadata_list={0: 1},
        _recv_attn_tensor_metadata_list={0: 1},
        _recv_attn_buffers={0: 1},
        _recv_attn_input_ids_buffers={0: 1},
    )
    namespace["close"](connector)
    namespace["close"](connector)
    assert events == ["a2f", "f2a", "control_group", "afd_world"]
    assert registry == {3: "another_connector"}
    assert connector.a2e_group is None and connector.e2a_group is None
    assert not connector._initialized
    assert not connector._recv_attn_buffers
