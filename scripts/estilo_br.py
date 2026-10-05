"""Formato brasileiro nas figuras: virgula decimal nos eixos e nos rotulos numericos."""
import re
import matplotlib.ticker as mt
from matplotlib.figure import Figure

_NUM = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w.])")
def _br(s):
    return _NUM.sub(lambda m: m.group().replace(".", ","), s)

class FormatoBR(mt.Formatter):
    """Envolve o formatador original e troca o ponto decimal por virgula."""
    _br = True
    def __init__(self, base):
        self.base = base
    def set_axis(self, axis):
        super().set_axis(axis); self.base.set_axis(axis)
    def set_locs(self, locs):
        self.base.set_locs(locs)
    def __call__(self, x, pos=None):
        return _br(self.base(x, pos))
    def format_ticks(self, values):
        return [_br(s) for s in self.base.format_ticks(values)]
    def get_offset(self):
        return _br(self.base.get_offset())

def _ajusta(fig):
    for ax in fig.axes:
        for eixo in (ax.xaxis, ax.yaxis):
            f = eixo.get_major_formatter()
            if not getattr(f, "_br", False):
                eixo.set_major_formatter(FormatoBR(f))

_salvar = Figure.savefig
def _savefig(self, *a, **k):
    _ajusta(self)
    return _salvar(self, *a, **k)
Figure.savefig = _savefig
