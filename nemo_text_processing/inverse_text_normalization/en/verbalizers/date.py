import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)


def _build_reorder_mmddyyyy_to_yyyymmdd():
    d = pynini.union(*[str(i) for i in range(10)])
    insert_sep = d + d + d + d + pynutil.insert("|") + d + d + d + d

    reorder_parts = []
    for m in range(1, 13):
        for day in range(1, 32):
            mmdd = f"{m:02d}{day:02d}"
            reorder_parts.append(
                pynini.cross(mmdd + "|", "") + d + d + d + d + pynutil.insert(mmdd)
            )

    reorder = pynini.union(*reorder_parts).optimize()
    return (insert_sep @ reorder).optimize()


class DateFst(GraphFst):

    def __init__(self):
        super().__init__(name="date", kind="verbalize")

        add_leading_zero = (NEMO_DIGIT + NEMO_DIGIT) | (pynutil.insert("0") + NEMO_DIGIT)

        month_to_number = (
            pynini.cross("january", "01") | pynini.cross("february", "02")
            | pynini.cross("march", "03") | pynini.cross("april", "04")
            | pynini.cross("may", "05") | pynini.cross("june", "06")
            | pynini.cross("july", "07") | pynini.cross("august", "08")
            | pynini.cross("september", "09") | pynini.cross("october", "10")
            | pynini.cross("november", "11") | pynini.cross("december", "12")
        )

        month_num = (
            pynutil.delete("month:") + delete_space + pynutil.delete("\"")
            + (pynini.closure(NEMO_NOT_QUOTE, 1) @ month_to_number)
            + pynutil.delete("\"")
        )

        day_padded = (
            pynutil.delete("day:") + delete_space + pynutil.delete("\"")
            + (pynini.closure(NEMO_DIGIT, 1) @ add_leading_zero)
            + pynutil.delete("\"")
        )

        year = (
            pynutil.delete("year:") + delete_space + pynutil.delete("\"")
            + pynini.closure(NEMO_DIGIT, 1)
            + delete_space + pynutil.delete("\"")
        )

        period = (
            pynutil.delete("text:") + delete_space + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        optional_preserve_order = pynini.closure(
            pynutil.delete("preserve_order:") + delete_space + pynutil.delete("true") + delete_space
            | pynutil.delete("field_order:") + delete_space
            + pynutil.delete("\"") + NEMO_NOT_QUOTE + pynutil.delete("\"") + delete_space
        )

        reorder = _build_reorder_mmddyyyy_to_yyyymmdd()

        graph_mdy = (month_num + delete_space + day_padded + delete_space + year) @ reorder

        graph_md = month_num + pynutil.insert("/") + delete_space + day_padded

        graph_dmy_raw = day_padded + delete_space + month_num + delete_space + year
        swap_ddmm = []
        for m in range(1, 13):
            for day in range(1, 32):
                swap_ddmm.append(pynini.cross(f"{day:02d}{m:02d}", f"{m:02d}{day:02d}"))
        swap_first4 = pynini.union(*swap_ddmm).optimize()
        d = pynini.union(*[str(i) for i in range(10)])
        swap_ddmm_pass_yyyy = (swap_first4 + d + d + d + d).optimize()
        graph_dmy = (graph_dmy_raw @ swap_ddmm_pass_yyyy) @ reorder

        graph_y = year
        graph_fy = period + pynini.closure(delete_extra_space + year, 0, 1)

        final_graph = (graph_mdy | graph_md | graph_dmy | graph_y | graph_fy) + delete_space + optional_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
