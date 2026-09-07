import subprocess
import sys
import pytest
from hsi.cli import parser


def test_help_does_not_import_models():
    code = 'from hsi.cli import parser; import sys; parser().format_help(); assert "torch" not in sys.modules; assert "comfy" not in sys.modules'
    subprocess.run([sys.executable, '-c', code], check=True)


@pytest.mark.parametrize('name', ['check-scene','scene-build','plan','views','image-job','image','recover',
    'stage2','compile-motion','motion','evaluate-motion'])
def test_stage_help(name):
    with pytest.raises(SystemExit) as result:
        parser().parse_args([name,'--help'])
    assert result.value.code == 0


def test_view_command_accepts_receipt_path(monkeypatch,tmp_path,capsys):
    from hsi.cli import main
    import hsi.stage2.views as views
    receipt=tmp_path/'receipt.json'
    monkeypatch.setattr(views,'select_views',lambda *args,**kwargs:receipt)
    assert main(['views','--stage1',str(tmp_path/'stage1.json'),'--render',str(tmp_path/'render.json'),
        '--output',str(tmp_path/'output')]) == 0
    assert str(receipt) in capsys.readouterr().out
