# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pynini
from pynini.lib import pynutil

from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_DIGIT,
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_space,
)


class DateFst(GraphFst):
    """
    Finite state transducer for verbalizing date, e.g.
        date { month: "january" day: "5" year: "2025" preserve_order: true } -> 01/05/2025
    """

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

        slash = pynutil.insert("/")

        # month/day/year -> MM/DD/YYYY
        graph_mdy = (
            month_num + slash
            + delete_space + day_padded + slash
            + delete_space + year
        )

        # month/day (no year) -> MM/DD
        graph_md = month_num + slash + delete_space + day_padded

        # day/month/year -> DD/MM/YYYY
        graph_dmy = (
            day_padded + slash
            + delete_space + month_num + slash
            + delete_space + year
        )

        graph_y = year
        graph_fy = period + pynini.closure(delete_extra_space + year, 0, 1)

        final_graph = (graph_mdy | graph_md | graph_dmy | graph_y | graph_fy) + delete_space + optional_preserve_order

        delete_tokens = self.delete_tokens(final_graph)
        self.fst = delete_tokens.optimize()
