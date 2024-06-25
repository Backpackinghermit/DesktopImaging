import cv2 as cv
import numpy as np
import os
import SimpleITK as sitk
import shutil


def perform_image_registration(vis_image_path, ir_image_path, output_path, image_type):
    try:
        print("Starting registration...")
        filename = os.path.join(output_path, f"registered_{image_type}_{os.path.basename(ir_image_path)}")
        
        # Load images
        img2 = cv.imread(vis_image_path, cv.IMREAD_GRAYSCALE)  # referenceImage
        img1 = cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE)  # sensedImage
        
        # Invert the color of the infrared image (img1)
        img1 = cv.bitwise_not(img1)

        # Resize the images
        resize_factor = 1.0 / 4.0
        img1_rs = cv.resize(img1, (0, 0), fx=resize_factor, fy=resize_factor)
        img2_rs = cv.resize(img2, (0, 0), fx=resize_factor, fy=resize_factor)

        # Initiate SIFT detector
        sift_detector = cv.SIFT_create()

        # Find the keypoints and descriptors with SIFT on the lower resolution images
        kp1, des1 = sift_detector.detectAndCompute(img1_rs, None)
        kp2, des2 = sift_detector.detectAndCompute(img2_rs, None)

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Filter out poor matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.35 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 20:  # Adjust the threshold as needed
            raise Exception("Insufficient keypoints for registration")

        matches = good_matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        H, mask = cv.findHomography(points1, points2, cv.RANSAC)

        # Get low-res and high-res sizes
        low_height, low_width = img1_rs.shape
        height, width = img1.shape
        low_size = np.float32([[0, 0], [0, low_height], [low_width, low_height], [low_width, 0]])
        high_size = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        # Compute scaling transformations
        scale_up = cv.getPerspectiveTransform(low_size, high_size)
        scale_down = cv.getPerspectiveTransform(high_size, low_size)

        # Combine the transformations. Remember that the order of the transformation
        # is reversed when doing matrix multiplication
        # so this is actually scale_down -> H -> scale_up
        h_and_scale_up = np.matmul(scale_up, H)
        scale_down_h_scale_up = np.matmul(h_and_scale_up, scale_down)

        # Warp image 1 to align with image 2
        img1Reg = cv.warpPerspective(
            img1,
            scale_down_h_scale_up,
            (img2.shape[1], img2.shape[0])
        )
        
        img1_registered = cv.bitwise_not(img1Reg)


        # Save the processed image
        print("Registration successful.")
        cv.imwrite(filename, img1_registered)
        return filename

    except Exception as e:
        print("Image registration failed, trying broader parameters:", e)
        return registration_broad(vis_image_path, ir_image_path, output_path, image_type) #try broad parameters


def registration_broad(vis_image_path, ir_image_path, output_path, image_type):
    try:
        print("Starting registration...")
        filename = os.path.join(output_path, f"registered_{image_type}_{os.path.basename(ir_image_path)}")
        # Load images
        # Load images
        img2 = cv.imread(vis_image_path, cv.IMREAD_GRAYSCALE)  # referenceImage
        img1 = cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE)  # sensedImage

        # Resize the images
        resize_factor = 1.0 / 4.0
        img1_rs = cv.resize(img1, (0, 0), fx=resize_factor, fy=resize_factor)
        img2_rs = cv.resize(img2, (0, 0), fx=resize_factor, fy=resize_factor)

        # Initiate SIFT detector
        sift_detector = cv.SIFT_create()

        # Find the keypoints and descriptors with SIFT on the lower resolution images
        kp1, des1 = sift_detector.detectAndCompute(img1_rs, None)
        kp2, des2 = sift_detector.detectAndCompute(img2_rs, None)

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Filter out poor matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.50 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 20:  # Adjust the threshold as needed
            raise Exception("Insufficient keypoints for registration")

        matches = good_matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt

        # Find homography
        H, mask = cv.findHomography(points1, points2, cv.RANSAC)

        # Get low-res and high-res sizes
        low_height, low_width = img1_rs.shape
        height, width = img1.shape
        low_size = np.float32([[0, 0], [0, low_height], [low_width, low_height], [low_width, 0]])
        high_size = np.float32([[0, 0], [0, height], [width, height], [width, 0]])

        # Compute scaling transformations
        scale_up = cv.getPerspectiveTransform(low_size, high_size)
        scale_down = cv.getPerspectiveTransform(high_size, low_size)

        # Combine the transformations. Remember that the order of the transformation
        # is reversed when doing matrix multiplication
        # so this is actually scale_down -> H -> scale_up
        h_and_scale_up = np.matmul(scale_up, H)
        scale_down_h_scale_up = np.matmul(h_and_scale_up, scale_down)

        # Warp image 1 to align with image 2
        img1Reg = cv.warpPerspective(
            img1,
            scale_down_h_scale_up,
            (img2.shape[1], img2.shape[0])
        )

        # Save the processed image
        cv.imwrite(filename, img1Reg)

        print("Registration successful with broader parameters.")
        return filename

    except Exception as e:
        print(f"Image registration failed with broader parameters for {image_type}:", e)
        print("Trying mutual information-based registration")

        return mutual_information_registration(vis_image_path, ir_image_path, output_path, image_type) #try broad parameters

def mutual_information_registration(vis_image_path, ir_image_path, output_path, image_type):
    try:
        print("Starting mutual information-based registration...")
        filename = os.path.join(output_path, f"registered_{image_type}_{os.path.basename(ir_image_path)}")

        # Load images
        img2 = cv.imread(vis_image_path, cv.IMREAD_GRAYSCALE)  # referenceImage
        img1 = cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE)  # sensedImage

        # Convert images to SimpleITK format and then to 32-bit float
        img1_sitk = sitk.GetImageFromArray(img1.astype(np.float32))
        img2_sitk = sitk.GetImageFromArray(img2.astype(np.float32))

        # Initialize registration
        registration_method = sitk.ImageRegistrationMethod()

        # Set the similarity metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=300)

        # Set the optimizer
        registration_method.SetOptimizerAsGradientDescent(learningRate=0.01, numberOfIterations=400, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

        # Set the initial transform
        initial_transform = sitk.CenteredTransformInitializer(img2_sitk, img1_sitk, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Perform the registration
        final_transform = registration_method.Execute(img2_sitk, img1_sitk)

        # Resample the image
        img1_registered_sitk = sitk.Resample(img1_sitk, img2_sitk, final_transform, sitk.sitkLinear, 0.0, img1_sitk.GetPixelID())

        # Convert back to numpy array
        img1_registered = sitk.GetArrayFromImage(img1_registered_sitk)

        # Save the processed image
        cv.imwrite(filename, img1_registered)

        print("Mutual information-based registration successful.")
        
        return filename

    except Exception as e:
        print("Mutual information-based registration failed:", e)
        return cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE)  # Return input image

def orb_registration(vis_image_path, ir_image_path, output_path):
        try:
            print("Starting ORB-based registration...")

            # Load images
            img2 = cv.imread(vis_image_path, cv.IMREAD_GRAYSCALE)  # referenceImage
            img1 = cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE)  # sensedImage

            # Initialize ORB detector with a higher number of features
            orb_detector = cv.ORB_create(nfeatures=1000)

            # Find keypoints and descriptors
            kp1, des1 = orb_detector.detectAndCompute(img1, None)
            kp2, des2 = orb_detector.detectAndCompute(img2, None)

            # Use BFMatcher to find the best matches
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)

            # Sort matches based on distance
            matches = sorted(matches, key=lambda x: x.distance)

            # Optionally, keep only the top matches
            num_good_matches = int(len(matches) * 0.15)  # Adjust this percentage as needed
            good_matches = matches[:num_good_matches]

            if len(good_matches) < 10:  # Adjust the threshold as needed
                raise Exception("Insufficient keypoints for registration")

            # Extract location of good matches
            points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
            points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

            for i, match in enumerate(good_matches):
                points1[i, :] = kp1[match.queryIdx].pt
                points2[i, :] = kp2[match.trainIdx].pt

            # Find homography with adjusted RANSAC parameters
            H, mask = cv.findHomography(points1, points2, cv.RANSAC, 5.0)  # Adjust RANSAC threshold

            # Warp image 1 to align with image 2
            img1Reg = cv.warpPerspective(
                img1,
                H,
                (img2.shape[1], img2.shape[0])
            )

            # Save the processed image
            cv.imwrite(os.path.join(output_path, 'registered_orb.jpg'), img1Reg)

            print("ORB-based registration successful.")
            return img1Reg

        except Exception as e:
            print("ORB-based registration failed:", e)
            return cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE)  # Return input image