import dpdata.rescuplus.scf
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
from dpdata.data_type import Axis, DataType
from dpdata.format import Format
from dpdata.utils import open_file

if TYPE_CHECKING:
    from dpdata.utils import FileType

@Format.register("rescuplus/scf")
class RescuplusSCFFormat(Format):
    # @Format.post("rot_lower_triangular")
    def from_labeled_system(self, file_name, **kwargs):
        data = dpdata.rescuplus.scf.get_frame(file_name)
        return data

@Format.register("rescuplus/md")
class RescuplusMDFormat(Format):
    # @Format.post("rot_lower_triangular")
    def from_labeled_system(self, file_name, **kwargs):
        data = dpdata.rescuplus.md.get_frame(file_name)
        return data
    
@Format.register("rescuplus/relax")
class RescuplusRELAXFormat(Format):
    # @Format.post("rot_lower_triangular")
    def from_labeled_system(self, file_name, **kwargs):
        data = dpdata.rescuplus.relax.get_frame(file_name)
        return data

