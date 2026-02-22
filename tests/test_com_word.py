from __future__ import annotations

from docxru.com_word import _autofit_textbox_shape


class _FakeFont:
    def __init__(self, size: float) -> None:
        self.Size = float(size)


class _FakeTextRange:
    def __init__(self, font: _FakeFont) -> None:
        self.Font = font


class _FakeTextFrame:
    def __init__(self, *, size: float, overflow_until_size: float) -> None:
        self.HasText = 1
        self.AutoSize = 0
        self._overflow_until_size = float(overflow_until_size)
        self.TextRange = _FakeTextRange(_FakeFont(size))

    @property
    def Overflowing(self) -> bool:  # noqa: N802
        return float(self.TextRange.Font.Size) > self._overflow_until_size


class _FakeTextFrame2:
    def __init__(self) -> None:
        self.AutoSize = 0


class _FakeShape:
    def __init__(self, text_frame: _FakeTextFrame | None) -> None:
        self.TextFrame = text_frame
        self.TextFrame2 = _FakeTextFrame2()
        self.Type = 1


def test_autofit_textbox_shape_shrinks_when_overflowing():
    shape = _FakeShape(_FakeTextFrame(size=11.0, overflow_until_size=8.5))

    stats = _autofit_textbox_shape(shape, min_font_size_pt=8.0, max_shrink_steps=3)

    assert stats["textboxes_seen"] == 1
    assert stats["textboxes_autofit"] == 1
    assert stats["textboxes_shrunk"] == 1
    assert shape.TextFrame.AutoSize == 1
    assert shape.TextFrame2.AutoSize == 1
    assert float(shape.TextFrame.TextRange.Font.Size) == 8.0


def test_autofit_textbox_shape_ignores_shapes_without_text():
    class _EmptyTextFrame:
        HasText = 0
        AutoSize = 0

    shape = _FakeShape(_EmptyTextFrame())  # type: ignore[arg-type]

    stats = _autofit_textbox_shape(shape, min_font_size_pt=8.0, max_shrink_steps=2)

    assert stats == {"textboxes_seen": 0, "textboxes_autofit": 0, "textboxes_shrunk": 0}
