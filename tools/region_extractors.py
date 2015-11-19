import numpy
import numpy as np
import logging
logger = logging.getLogger('digits.tools.region_extractors')

class BoundingBoxExtractor:

    def __init__(self, scale_factor=None, resize_shape=None):
        """
        part_id : the part id of the bbox in the model data
        scale_factor : a factor to enlarge the bounding box by. The center of the bounding box will be maintained
        resize_shape : a (height, width) tuple to resize the bounding box to
        """

        self.scale_factor = scale_factor
        self.resize_shape = resize_shape


    def extract(self, image, bbox):
        """
        image -- a numpy ndarray that represents the image, must be at least 2d
        bbox -- patch in image to extract [x,y,w,h] where x and y are the top left corner
        """

        bbox_x, bbox_y, bbox_width, bbox_height = bbox
        image_width, image_height = image.size
        bbox_x = int(max(0, bbox_x))
        bbox_y = int(max(0, bbox_y))
        bbox_width = int(min(image_width - bbox_x, bbox_width))
        bbox_height = int(min(image_height - bbox_y, bbox_height))

        return self.extract_bbox(image, bbox_x, bbox_y, bbox_width, bbox_height)


    def extract_bbox(self, image, bbox_x, bbox_y, bbox_width, bbox_height):
        """
        image -- a numpy ndarray that represents the image must, be at least 2d
        bbox_x -- (int) the column index of the upper left corner of the bounding box, in image coordinates.
        bbox_y -- (int) the row index of the upper left corner of the bounding box, in image coordinates.
        bbox_width -- (int) the width of the bbox, in image coordinates
        bbox_height -- (int) the height of the bbox, in image coordinates
        """

        if self.scale_factor is not None:

            image_width, image_height = image.size
            scale_factor = self.scale_factor
            half_width = bbox_width / 2
            half_height = bbox_height / 2

            bbox_center_x = bbox_x + half_width
            bbox_center_y = bbox_y + half_height

            # we want the object to stay in the middle, so the expansion has to be the same in both directions for each axis.
            width_expansion = min(bbox_center_x - max(int(bbox_center_x - scale_factor * half_width), 0), min(int(bbox_center_x + scale_factor * half_width), image_width) - bbox_center_x)
            height_expansion = min(bbox_center_y - max(int(bbox_center_y - scale_factor * half_height), 0), min(int(bbox_center_y + scale_factor * half_height), image_height) - bbox_center_y)

            bbox_x1 = bbox_center_x - width_expansion
            bbox_x2 = bbox_center_x + width_expansion
            bbox_y1 = bbox_center_y - height_expansion
            bbox_y2 = bbox_center_y + height_expansion

        else:
            bbox_x1 = bbox_x
            bbox_x2 = bbox_x + bbox_width
            bbox_y1 = bbox_y
            bbox_y2 = bbox_y + bbox_height

        crop = image.crop((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        crop.load()

        #if self.resize_shape is not None:
        #    crop = transform.resize(crop, self.resize_shape)
        #    crop = crop * 255

        return crop

