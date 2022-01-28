import cv2
import easyocr
# import keras_ocr
import numpy as np
import pytesseract
import tesserocr as tr

from PIL import Image
from rich import print


class ImageProcessor:
    def __init__(self, image) -> None:
        self.image = image

    def ocr_1(self) -> str:
        """generate text from image using tesseract ocr

        :return: text from image
        :rtype: str
        """
        img = cv2.imread(self.image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite("src/files/filterImg.png", th1)

        pilImg = Image.open("src/files/filterImg.png")
        text = pytesseract.image_to_string(pilImg)
        return str(text.encode("utf-8")).lower()

    def ocr_2(self) -> str:
        """generate text from image using easyocr

        :return: text from image
        :rtype: str
        """
        reader = easyocr.Reader(["en"], gpu=False)
        img = cv2.imread(self.image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        texts = reader.readtext(gray, detail=0)
        return str(texts).lower()

    # def ocr_3(self) -> str:
    #     """generate text from image using keras_ocr

    #     :return: text from image
    #     :rtype: str
    #     """
    #     pipeline = keras_ocr.pipeline.Pipeline()
    #     image = keras_ocr.tools.read(self.image)
    #     prediction_groups = pipeline.recognize([image])[0]
    #     return str([text for text, box in prediction_groups])

    def ocr_4(self) -> str:
        number_ok = cv2.imread(self.image)
        blur = cv2.medianBlur(number_ok, 1)

        pil_img = Image.fromarray(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))

        api = tr.PyTessBaseAPI()

        try:
            api.SetImage(pil_img)
            text = api.GetUTF8Text()
        finally:
            api.End()
        return str(text).lower()

    def ocr_5(self) -> str:
        img = cv2.imread(self.image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        invGamma = 1.0 / 0.3
        table = np.array(
            [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # apply gamma correction using the lookup table
        gray = cv2.LUT(gray, table)

        ret, thresh1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(
            thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )[-2:]

        def biggestRectangle(contours):
            biggest = None
            max_area = 0
            indexReturn = -1
            for index in range(len(contours)):
                i = contours[index]
                area = cv2.contourArea(i)
                if area > 100:
                    peri = cv2.arcLength(i, True)
                    approx = cv2.approxPolyDP(i, 0.1 * peri, True)
                    if area > max_area:  # and len(approx)==4:
                        biggest = approx
                        max_area = area
                        indexReturn = index
            return indexReturn

        indexReturn = biggestRectangle(contours)
        hull = cv2.convexHull(contours[indexReturn])

        # create a crop mask
        mask = np.zeros_like(
            img
        )  # Create mask where white is what we want, black otherwise
        cv2.drawContours(
            mask, contours, indexReturn, 255, -1
        )  # Draw filled contour in mask
        out = np.zeros_like(img)  # Extract out the object and place into output image
        out[mask == 255] = img[mask == 255]

        # crop the image
        (y, x, _) = np.where(mask == 255)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))
        out = img[topy : bottomy + 1, topx : bottomx + 1, :]

        # predict tesseract
        lang = "eng+nld"
        config = "--psm 11 --oem 3"
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

        img_data = pytesseract.image_to_data(
            out_rgb,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DATAFRAME,
        )
        img_conf_text = img_data[["conf", "text"]]
        img_valid = img_conf_text[img_conf_text["text"].notnull()]
        img_words = img_valid[img_valid["text"].str.len() > 1]

        all_predictions = img_words["text"].to_list()

        return f"{all_predictions}".lower()

    def mse(self, imageA, imageB):
        # the 'Mean Squared Error' between the two images is the
        # sum of the squared difference between the two images;
        # NOTE: the two images must have the same dimension
        def prepare_img(im):
            size = 300, 200
            im = cv2.resize(im, size)
            return im

        imageA = cv2.imread(imageA)
        imageB = cv2.imread(imageB)
        err = np.sum(
            (prepare_img(imageA).astype("float") - prepare_img(imageB).astype("float"))
            ** 2
        )
        err /= float(imageA.shape[0] * imageA.shape[1])

        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err

    def image_type(self):
        image_passport = 0
        image_id = 0
        test1 = self.ocr_1()
        test2 = self.ocr_2()
        # test3 = self.ocr_3()
        test4 = self.ocr_4()
        test5 = self.ocr_5()
        # print(test1, "\n", test2, "\n", test3, "\n", test4, "\n", test5)
        print(test1, "\n", test2, "\n", test4, "\n", test5)
        if "passport" in test1:
            image_passport += 1
        if "passport" in test2:
            image_passport += 1
        # if "passport" in test3:
        #     image_passport += 1
        if "passport" in test4:
            image_passport += 1
        if "passport" in test5:
            image_passport += 1
        if "id" in test1 and "number" in test1:
            image_id += 1
        if "id" in test2 and "number" in test2:
            image_id += 1
            return 'ID Card'
        # if "id" in test3:
        #     image_id += 1
        if "id" in test4 and "number" in test4:
            image_id += 1
        if "id" in test5 and "number" in test4:
            image_id += 1
        if image_id > 1:
            return "ID card"
        elif image_passport > 1:
            return "Passport Card"
        return "NOT a passport or an ID card"
