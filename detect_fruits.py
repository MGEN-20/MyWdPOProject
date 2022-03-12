import json
from pathlib import Path
from typing import Dict
import numpy as np

import click
import cv2
from tqdm import tqdm
from glob import glob


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """

    #TODO: Implement detection method.
    apple = 0
    banana = 0
    orange = 0

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.medianBlur(img, 7)
    img = cv2.resize(img, (520, 420), interpolation=cv2.INTER_AREA)
    copy = img.copy()

    hsv_frame = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
    # orange

    low_orange = np.array([0, 190, 173])
    high_orange = np.array([20, 255, 254])
    mask1 = cv2.inRange(hsv_frame, low_orange, high_orange)
    res_org = cv2.bitwise_and(copy, copy, mask=mask1)
    res_bgr_or = cv2.cvtColor(res_org, cv2.COLOR_HSV2BGR)

    # banana

    low_banana = np.array([18, 81, 138])
    high_banana = np.array([138, 255, 253])
    mask2 = cv2.inRange(hsv_frame, low_banana, high_banana)
    res_ban = cv2.bitwise_and(copy, copy, mask=mask2)
    res_bgr_ban = cv2.cvtColor(res_ban, cv2.COLOR_HSV2BGR)

    # apple
    low_apple1 = np.array([146, 52, 0])
    high_apple1 = np.array([254, 222, 255])
    low_apple2 = np.array([0, 52, 0])
    high_apple2 = np.array([14, 222, 255])

    mask3_1 = cv2.inRange(hsv_frame, low_apple1, high_apple1)
    mask3_2 = cv2.inRange(hsv_frame, low_apple2, high_apple2)

    mask3 = cv2.bitwise_or(mask3_1, mask3_2)
    res_ap = cv2.bitwise_and(copy, copy, mask=mask3)
    res_bgr_ap = cv2.cvtColor(res_ap, cv2.COLOR_HSV2BGR)

    orange
    res_gray_or = cv2.cvtColor(res_bgr_or, cv2.COLOR_BGR2GRAY)
    dilation_org = cv2.dilate(res_gray_or, (21, 21), iterations=1)
    contours_org, _ = cv2.findContours(dilation_org, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(copy, contours_org, -1, (0, 0, 255), 4)
    for contour in contours_org:
        if cv2.contourArea(contour) > 1000:
            orange += 1

    # banana
    res_gray_ban = cv2.cvtColor(res_bgr_ban, cv2.COLOR_BGR2GRAY)
    dilation_ban = cv2.dilate(res_gray_ban, (21, 21), iterations=1)
    contours_ban, _ = cv2.findContours(dilation_ban, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(copy, contours_ban, -1, (0, 255, 0), 4)
    for contour in contours_ban:
        if cv2.contourArea(contour) > 1000:
            banana += 1

    # apple
    res_gray_ap = cv2.cvtColor(res_bgr_ap, cv2.COLOR_BGR2GRAY)
    dilation_app = cv2.dilate(res_gray_ap, (21, 21), iterations=1)
    contours_ap, _ = cv2.findContours(dilation_app, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(copy, contours_ap, -1, (255, 0, 0), 4)
    for contour in contours_ap:
        if cv2.contourArea(contour) > 900:
            apple += 1
    cv2.imshow('img',copy)
    cv2.waitKey(0)

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits
        print('\n {}'.format(fruits))

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
