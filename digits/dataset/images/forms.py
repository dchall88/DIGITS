# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import wtforms
from wtforms import validators

from ..forms import DatasetForm
from job import ImageDatasetJob
from digits import utils
from digits.utils.forms import validate_bounding_box_with_json

class ImageDatasetForm(DatasetForm):
    """
    Defines the form used to create a new ImageDatasetJob
    (abstract class)
    """

    encoding = utils.forms.SelectField('Image Encoding',
            default = 'png',
            choices = [
                ('none', 'None'),
                ('png', 'PNG (lossless)'),
                ('jpg', 'JPEG (lossy, 90% quality)'),
                ],
            tooltip = "Using either of these compression formats can save disk space, but can also require marginally more time for training."
            )

    ### Image resize

    resize_channels = utils.forms.SelectField(u'Image Type',
            default='3',
            choices=[('1', 'Grayscale'), ('3', 'Color')],
            tooltip = "Color is 3-channel RGB. Grayscale is single channel monochrome."
            )
    resize_width = wtforms.IntegerField(u'Resize Width',
            default=256,
            validators=[validators.DataRequired()]
            )
    resize_height = wtforms.IntegerField(u'Resize Height',
            default=256,
            validators=[validators.DataRequired()]
            )
    resize_mode = utils.forms.SelectField(u'Resize Transformation',
            default='squash',
            choices=ImageDatasetJob.resize_mode_choices(),
            tooltip = "Options for dealing with aspect ratio changes during resize. See examples below."
            )

    ### Extract Bounding Boxes
    bbox_mode = utils.forms.SelectField(u'Extract Bounding Boxes',
            default='0',
            choices=[
                ('0', 'No'),
                ('1', 'Yes'),
                ],
            validators=[validate_bounding_box_with_json()],
            tooltip="Extract regions from the image defined by bounding boxes."
            )

    scale_factor = utils.forms.FloatField(u'Scale Factor',
            default=1.0,
            validators=[validators.DataRequired(), validators.NumberRange(min=1),],
            tooltip="Add uniform padding to the bounding box by specifying a scale factor"
            )