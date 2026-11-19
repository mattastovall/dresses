import unittest

import numpy as np
from PIL import Image

from dressflow.retouch import PortraitMaterial, inverse_map


class RetouchTests(unittest.TestCase):
    def test_new_fingers_stay_in_front_of_old_garment(self):
        w, h = 150, 180
        yy, xx = np.indices((h, w))
        original = np.full((h, w, 3), 175, dtype="uint8")
        cloth = (xx > 25) & (xx < 125) & (yy > 18) & (yy < 169)
        # Shading across a chromatic textile gives the transfer real midtones.
        shade = ((xx + yy) % 61)[cloth]
        original[cloth] = np.column_stack(
            (165 + shade, 15 + shade // 3, 72 + shade // 2)
        )
        portrait = original.copy()
        hand = (xx >= 59) & (xx <= 67) & (yy < 112) & cloth
        # A strong pink reflection on a finger must still be protected.
        portrait[hand] = [203, 77, 110]
        visible = cloth & ~hand
        points = [
            [point, point]
            for point in (
                [32, 25],
                [76, 28],
                [117, 30],
                [34, 151],
                [73, 159],
                [115, 150],
            )
        ]
        mx, my = inverse_map((w, h), points)
        np.testing.assert_allclose(mx, xx)
        np.testing.assert_allclose(my, yy)
        material = PortraitMaterial(
            Image.fromarray(portrait),
            Image.fromarray(original),
            Image.fromarray(cloth.astype("uint8") * 255),
            Image.fromarray(visible.astype("uint8") * 255),
            [[26, 19], [124, 19], [124, 168], [26, 168]],
            points,
        )
        old_render = original.copy()
        old_render[cloth] = [38, 52, 78]
        revised, _ = material.apply(Image.fromarray(old_render), [38, 52, 78])
        pixels = np.asarray(revised)
        # The same shared portrait supplies every visible hand, including its edge cleanup.
        np.testing.assert_array_equal(pixels[~visible], material.portrait[~visible])
        np.testing.assert_array_equal(pixels[65, 63], portrait[65, 63])
        np.testing.assert_array_equal(pixels[140, 90], old_render[140, 90])
        self.assertFalse(material.panel_weight[hand].any())


if __name__ == "__main__":
    unittest.main()
