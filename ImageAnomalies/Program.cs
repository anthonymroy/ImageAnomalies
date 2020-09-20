using System;
using System.Drawing;
using System.IO;
using OpenCvSharp;
using OpenCvSharp.ML;

namespace ImageAnomalies
{
    class Program
    {
        static readonly string InputDirectory = "../../../../Images/Input/";
        static readonly string OutputDirectory = "../../../../Images/Output/";
        static readonly string MaskDirectory = OutputDirectory + "Masks/";
        static void Main(string[] args)
        {
            string errMsg = CheckDirectories();
            if (errMsg != string.Empty)
            {
                Console.Write(errMsg);
                return;
            }

            string[] inputFiles = Directory.GetFiles(InputDirectory, "*hr.bmp");
            Mat[] inputImages = new Mat[inputFiles.Length];
            for (int i = 0; i<inputFiles.Length; i++)
            {
                inputImages[i] = new Mat(inputFiles[i], ImreadModes.Color);
            }
            OpenCvSharp.Size originalSize = inputImages[0].Size();                       

            inputImages = ConvertToGreyscale(inputImages);
            inputImages = Resize(inputImages, 0.125);
            inputImages = AutoContrast(inputImages);

            Mat goldenImage = GenerateGoldenImage(inputImages, 4);
            goldenImage.ConvertTo(goldenImage, MatType.CV_8UC1);
            Cv2.ImWrite(OutputDirectory + "golden_image.png", goldenImage);

            Mat ROIMask = GenerateROIMask(goldenImage);
            ROIMask.ConvertTo(ROIMask, MatType.CV_8UC1);
            Cv2.ImWrite(OutputDirectory + "ROI_mask.png", ROIMask);

            Mat[] masks = GenerateMasksFromGoldenImage(inputImages, goldenImage);

            Mat[] outputMasks = ApplyROIMask(masks, ROIMask);
            outputMasks = Resize(outputMasks, originalSize);
            SaveMasks(outputMasks);

        }

        static string CheckDirectories()
        {
            string errorMessage = string.Empty;

            if (!Directory.Exists(InputDirectory))
            {
                errorMessage += "Input directory: " + InputDirectory + " does not exist." + Environment.NewLine;
            }
            if (!Directory.Exists(OutputDirectory))
            {
                errorMessage += "Output directory: " + OutputDirectory + " does not exist." + Environment.NewLine;
            }
            if (!Directory.Exists(MaskDirectory))
            {
                errorMessage += "Mask directory: " + MaskDirectory + " does not exist." + Environment.NewLine;
            }
            return errorMessage;
        }

        static void SaveMasks(Mat[] masks)
        {
            for (int i = 0; i < masks.Length; i++)
            {                
                string filename = "mask" + (i+1).ToString("D") + ".png";
                Cv2.ImWrite(MaskDirectory + filename, masks[i]);
            }
        }
        ///<summary>
        ///Sets everything outside of the mask to 0
        ///</summary>
        static Mat[] ApplyROIMask(Mat[] images, Mat mask)
        {
            Mat[] output = new Mat[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                Mat img = images[i].EmptyClone();
                images[i].CopyTo(img, mask);
                output[i] = img;
            }
            return output;
        }
        ///<summary>
        ///Rescales images such that the minimumal value is 0 and maximum value is 255
        ///</summary>
        static Mat[] AutoContrast(Mat[] images)
        {
            Mat[] output = new Mat[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                Mat img = images[i].Clone();
                Cv2.MinMaxIdx(img, out double min, out double max);
                img = img - min;
                double range = max - min;
                img = img * 255 / range;
                output[i] = img;
            }
            return output;
        }
        ///<summary>
        ///Takes the composite of the first n images to create a golden image
        ///</summary>
        static Mat GenerateGoldenImage(Mat[] images, int numberOfImagesToUse = -1)
        {
            Mat goldenImage = images[0].EmptyClone();
            if (numberOfImagesToUse < 0)
            {
                numberOfImagesToUse = images.Length;
            }
            double scale = 1.0 / numberOfImagesToUse;
            for (int i=0; i< numberOfImagesToUse; i++)
            {
                Mat img = images[i].Clone();
                Cv2.ScaleAdd(img, scale, goldenImage, goldenImage);
            }
            return goldenImage;
        }
        ///<summary>
        ///Subtracts each image from the golden image and thresholds the difference
        ///</summary>
        static Mat[] GenerateMasksFromGoldenImage(Mat[] images, Mat goldenImage)
        {
            MatType type = MatType.CV_8UC1;
            if (images[0].Channels() > 1)
            {
                type = MatType.CV_8UC3;
            }
            Mat[] output = new Mat[images.Length];
            goldenImage.ConvertTo(goldenImage, type);
            Cv2.ImWrite(OutputDirectory + "golden_image.png", goldenImage);
            for (int i = 0; i < images.Length;i++)
            {
                Mat img = images[i].Clone();
                Mat diff_img = images[i].EmptyClone();
                Mat mask = images[i].EmptyClone();
                Cv2.Absdiff(img, goldenImage, diff_img);
                Cv2.Threshold(diff_img, mask, 32, 255, ThresholdTypes.Binary);
                output[i] = mask;                
            }
            return output;
        }

        static Mat[] Resize(Mat[] images, OpenCvSharp.Size size, double scale)
        {
            Mat[] output = new Mat[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                output[i] = images[i].EmptyClone();
                Cv2.Resize(images[i], output[i], size, scale, scale, InterpolationFlags.Cubic);
            }
            return output;
        }
        static Mat[] Resize(Mat[] images, double scale)
        {
            return Resize(images, new OpenCvSharp.Size(0, 0), scale);
        }

        static Mat[] Resize(Mat[] images, OpenCvSharp.Size size)
        {
            return Resize(images, size, 0);
        }

        static Mat[] ConvertToGreyscale(Mat[] images)
        {
            Mat[] output = new Mat[images.Length];
            for (int i = 0; i < images.Length; i++)
            {
                output[i] = images[i].EmptyClone();
                Cv2.CvtColor(images[i], output[i], ColorConversionCodes.BGR2GRAY);
            }
            return output;
        }
        ///<summary>
        ///Uses Hough lines to segment and extract the spacer grid 
        ///</summary>
        static Mat GenerateROIMask(Mat goldenImage)
        {
            Mat img = goldenImage.EmptyClone();
            Cv2.Blur(goldenImage, img, new OpenCvSharp.Size(15, 15));
            Cv2.Canny(img, img, 12, 24);

            LineSegmentPoint[] segHoughP = Cv2.HoughLinesP(img, 1, Math.PI / 180, 50, 5, 50);
            Mat mask = img.EmptyClone();

            foreach (LineSegmentPoint s in segHoughP)
            {
                mask.Line(s.P1, s.P2, Scalar.White, 1, LineTypes.AntiAlias, 0);
            }
            mask = DilateErode(mask, 7);
            Cv2.Threshold(mask, mask, 125, 255, ThresholdTypes.BinaryInv);
            return mask;
        }

        ///<summary>
        ///Dilate image n times then erode n-1 times to remove small bits of black space
        ///</summary>
        static Mat DilateErode(Mat img, int numberOfIterations)            
        {
            Mat mat = img.Clone();
            for (int i = 0; i < numberOfIterations; i++)
            {
                Cv2.Dilate(mat, mat, new Mat());
            }
            for (int i = 0; i < numberOfIterations - 1; i++)
            {
                Cv2.Erode(mat, mat, new Mat());
            }
            return mat;
        }
        ///<summary>
        ///Erode image n times then dilate an equal number of times to remove small bits of white space
        ///</summary>
        static Mat ErodeDilate(Mat img, int numberOfIterations)
        {
            Mat mat = img.Clone();
            for (int i = 0; i < numberOfIterations; i++)
            {
                Cv2.Erode(mat, mat, new Mat());
            }
            for (int i = 0; i < numberOfIterations; i++)
            {
                Cv2.Dilate(mat, mat, new Mat());
            }            
            return mat;
        }

    }
}
