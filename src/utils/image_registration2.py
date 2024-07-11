import cv2 as cv
import numpy as np
import os
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim



def perform_image_registration(vis_image_path, ir_image_path, output_path, image_type, resize_scale=8.0, match_space=0.2):
    if resize_scale <= 3.0:
        print("Switching to broader parameters.")
        return registration_broad(vis_image_path, ir_image_path, output_path, image_type)

    try:
        print(f"Starting registration with resize_scale={resize_scale} and match_space={match_space}...")
        filename = os.path.join(output_path, f"SIFT_registered_{image_type}_{os.path.basename(ir_image_path)}")
        
        # Load images
        img2 = cv.imread(vis_image_path, cv.IMREAD_GRAYSCALE)  # referenceImage
        img1 = cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE)  # sensedImage

        if img1 is None or img2 is None:
            raise Exception("Failed to load images.")

        # Resize the images
        resize_factor = 1.0 / resize_scale
        img1_rs = cv.resize(img1, (0, 0), fx=resize_factor, fy=resize_factor)
        img2_rs = cv.resize(img2, (0, 0), fx=resize_factor, fy=resize_factor)

        # Initiate SIFT detector
        sift_detector = cv.SIFT_create(nfeatures=200, contrastThreshold=0.04, edgeThreshold=5)

        # Find the keypoints and descriptors with SIFT on the lower resolution images
        kp1, des1 = sift_detector.detectAndCompute(img1_rs, None)
        kp2, des2 = sift_detector.detectAndCompute(img2_rs, None)

        if des1 is None or des2 is None:
            raise Exception("Failed to detect descriptors.")

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Filter out poor matches
        good_matches = []
        for m, n in matches:
            if m.distance < match_space * n.distance:
                good_matches.append(m)

        print(f"Found {len(good_matches)} good matches")

        if len(good_matches) < 20:  # Adjust the threshold as needed
            raise Exception("Insufficient keypoints for registration")

        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
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

        # Combine the transformations
        h_and_scale_up = np.matmul(scale_up, H)
        scale_down_h_scale_up = np.matmul(h_and_scale_up, scale_down)

        # Warp image 1 to align with image 2
        img1Reg = cv.warpPerspective(img1, scale_down_h_scale_up, (img2.shape[1], img2.shape[0]))
        
        #img1_registered = cv.bitwise_not(img1Reg)

        # Save the processed image
        cv.imwrite(filename, img1Reg)
        print("Registration successful.")
        return filename

    except Exception as e:
        print(f"Image registration failed, adjusting parameters: {e}")
        return perform_image_registration(vis_image_path, ir_image_path, output_path, image_type, resize_scale - 1, match_space + 0.1)

def registration_broad(vis_image_path, ir_image_path, output_path, image_type, match_space=0.4):
    if match_space >= 0.8:
        print("Failed to register the image with broader parameters. Trying MI")
        return mutual_information_registration(vis_image_path, ir_image_path, output_path, image_type)

    try:
        print(f"Starting registration2 with match_space={match_space}...")
        filename = os.path.join(output_path, f"SIFT_registered_{image_type}_{os.path.basename(ir_image_path)}")
        
        # Load images
        img2 = cv.imread(vis_image_path, cv.IMREAD_GRAYSCALE)  # referenceImage
        img1 = cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE)  # sensedImage

        if img1 is None or img2 is None:
            raise Exception("Failed to load images.")

        # Resize the images
        resize_factor = 1.0 / 4.0
        img1_rs = cv.resize(img1, (0, 0), fx=resize_factor, fy=resize_factor)
        img2_rs = cv.resize(img2, (0, 0), fx=resize_factor, fy=resize_factor)

        # Initiate SIFT detector
        sift_detector = cv.SIFT_create(nfeatures=200, contrastThreshold=0.04, edgeThreshold=5)

        # Find the keypoints and descriptors with SIFT on the lower resolution images
        kp1, des1 = sift_detector.detectAndCompute(img1_rs, None)
        kp2, des2 = sift_detector.detectAndCompute(img2_rs, None)

        if des1 is None or des2 is None:
            raise Exception("Failed to detect descriptors.")

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Filter out poor matches
        good_matches = []
        for m, n in matches:
            if m.distance < match_space * n.distance:
                good_matches.append(m)

        print(f"Found {len(good_matches)} good matches")

        if len(good_matches) < 20:  # Adjust the threshold as needed
            raise Exception("Insufficient keypoints for registration")

        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

        for i, match in enumerate(good_matches):
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

        # Combine the transformations
        h_and_scale_up = np.matmul(scale_up, H)
        scale_down_h_scale_up = np.matmul(h_and_scale_up, scale_down)

        # Warp image 1 to align with image 2
        img1Reg = cv.warpPerspective(img1, scale_down_h_scale_up, (img2.shape[1], img2.shape[0]))

        #img1_registered = cv.bitwise_not(img1Reg)

        # Save the processed image
        cv.imwrite(filename, img1Reg)
        print("Registration2 successful.")
        return filename

    except Exception as e:
        print(f"Image registration2 failed, adjusting parameters: {e}")
        return registration_broad(vis_image_path, ir_image_path, output_path, image_type, match_space + 0.05)
        

def mutual_information_registration(vis_image_path, ir_image_path, output_path, image_type, learning_rate=0.9, max_iterations=100, sampling_percentage=0.01):
    if max_iterations >= 500 or sampling_percentage > 1.0:
        print("MI failed")
        return mutual_information_registration2(vis_image_path, ir_image_path, output_path, image_type)

    try:
        print(f"Starting MI with learning_rate={learning_rate}, max_iterations={max_iterations}, sampling_percentage={sampling_percentage}")
        filename = os.path.join(output_path, f"MI1_registered_{image_type}_{os.path.basename(ir_image_path)}")

        # Load and preprocess images
        fixed_image = sitk.ReadImage(vis_image_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(ir_image_path, sitk.sitkFloat32)

        # Normalize images to the range [0, 1]
        fixed_image = sitk.RescaleIntensity(fixed_image, 0.0, 1.0)
        moving_image = sitk.RescaleIntensity(moving_image, 0.0, 1.0)

        # Invert the IR image before registration if necessary
        inverted_moving_image = sitk.InvertIntensity(moving_image, maximum=1.0)

        # Initialize registration
        registration_method = sitk.ImageRegistrationMethod()

        # Set the similarity metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(sampling_percentage)

        # Set the optimizer
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=learning_rate,
                                                                     minStep=1e-4,
                                                                     numberOfIterations=max_iterations,
                                                                     gradientMagnitudeTolerance=1e-4)

        # Set the interpolator
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Initial transform
        initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
        registration_method.SetInitialTransform(initial_transform)

        # Execute registration
        initial_metric_value = registration_method.GetMetricValue()
        final_transform = registration_method.Execute(fixed_image, inverted_moving_image)

        # Get the final metric value and check the threshold
        final_metric_value = registration_method.GetMetricValue()
        
        METRIC_IMPROVEMENT = 0.05
        print(f"Initial metric value: {initial_metric_value}")
        print(f"Final metric value: {final_metric_value}")
        print(f"Metric improvement: {final_metric_value - initial_metric_value}")
        
        if (final_metric_value - initial_metric_value) > METRIC_IMPROVEMENT:
            raise ValueError(f"Registration failed: Metric value improvement too low ({final_metric_value - initial_metric_value:.3f} < {METRIC_IMPROVEMENT:.3f})")

        # Resample the original (non-inverted) moving image
        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
        # Convert back to numpy array for OpenCV saving
        img1_registered = sitk.GetArrayFromImage(resampled_image)

        # Save the processed image
        cv.imwrite(filename, img1_registered)

        print(f"Mutual information-based registration successful. Output saved to {filename}")
        return filename
    
    except Exception as e:
        print("Mutual information-based registration failed:", e)
        return mutual_information_registration(vis_image_path, ir_image_path, output_path, image_type, learning_rate - 0.1, max_iterations + 100, min(sampling_percentage * 5, 1.0))

def registration_broad2(vis_image_path, ir_image_path, output_path, image_type):
    try:
        print("Starting SIFT2 registration...")
        filename = os.path.join(output_path, f"SIFT2_registered_{image_type}_{os.path.basename(ir_image_path)}")
        
        # Load images
        img2 = cv.imread(vis_image_path, cv.IMREAD_GRAYSCALE)  # referenceImage
        img1 = cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE)  # sensedImage

        # Resize the images
        resize_factor = 1.0 / 4.0
        img1_rs = cv.resize(img1, (0, 0), fx=resize_factor, fy=resize_factor)
        img2_rs = cv.resize(img2, (0, 0), fx=resize_factor, fy=resize_factor)

        # Initiate SIFT detector
        sift_detector = cv.SIFT_create()
        # Initiate SIFT detector with adjusted parameters
        sift_detector = cv.SIFT_create(nfeatures=500, contrastThreshold=0.02, edgeThreshold=10)

        # Find the keypoints and descriptors with SIFT on the lower resolution images
        kp1, des1 = sift_detector.detectAndCompute(img1_rs, None)
        kp2, des2 = sift_detector.detectAndCompute(img2_rs, None)

        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Filter out poor matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 30:  # Adjust the threshold as needed
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

        # Combine the transformations
        h_and_scale_up = np.matmul(scale_up, H)
        scale_down_h_scale_up = np.matmul(h_and_scale_up, scale_down)

        # Warp image 1 to align with image 2
        img1Reg = cv.warpPerspective(
            img1,
            scale_down_h_scale_up,
            (img2.shape[1], img2.shape[0])
        )

        # Save the processed image using OpenCV
        cv.imwrite(filename, img1Reg)

        # Reload the saved image using SimpleITK for further processing
        img1Reg_sitk = sitk.ReadImage(filename, sitk.sitkFloat32)

        print("Registration successful with registration_broad2.")
        return mutual_information_registration2(vis_image_path, filename, output_path, image_type)

    except Exception as e:
        print(f"SIFT2 failed: {e}")
        # Save the input image to the output directory with a prefix
        unregistered_filename = os.path.join(output_path, f"unregistered_{image_type}_{os.path.basename(ir_image_path)}")
        cv.imwrite(unregistered_filename, cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE))  # Save the original IR image
        return unregistered_filename


def mutual_information_registration2(vis_image_path, registered_filename, output_path, image_type):
    try:
        print("Starting mutual information-based registration2...")
        output_filename = os.path.join(output_path, f"registered_{image_type}_{os.path.basename(registered_filename)}")

        # Load and preprocess images using SimpleITK
        fixed_image = sitk.ReadImage(vis_image_path, sitk.sitkFloat32)
        moving_image = sitk.ReadImage(registered_filename, sitk.sitkFloat32)

        # Invert the IR image before registration
        inverted_moving_image = sitk.InvertIntensity(moving_image, maximum=255)

        # Initialize registration
        registration_method = sitk.ImageRegistrationMethod()

        # Set the similarity metric (experiment with alternatives)
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.6)

        # Set the optimizer (experiment with settings)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=0.01,
                                                                     minStep=1e-3,
                                                                     numberOfIterations=300,
                                                                     gradientMagnitudeTolerance=1e-5)

        # Set the interpolator (linear for good results)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Initial transform (using an affine transform for more flexibility)
        initial_transform = sitk.AffineTransform(fixed_image.GetDimension())
        registration_method.SetInitialTransform(initial_transform, inPlace=False)

        # Execute registration using the inverted moving image
        final_transform = registration_method.Execute(fixed_image, inverted_moving_image)

        # Resample the original (non-inverted) moving image
        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

        # Cast the resampled image to unsigned char for saving as JPEG
        resampled_image_cast = sitk.Cast(resampled_image, sitk.sitkUInt8)

        # Save the final registered image
        sitk.WriteImage(resampled_image_cast, output_filename)

        print(f"Mutual information-based registration2 successful. Output saved to {output_filename}")
        return output_filename
    
    except Exception as e:
        print("Mutual information-based registration2 failed:", e)

        # Save the input image to the output directory with a prefix
        unregistered_filename = os.path.join(output_path, f"unregistered_{image_type}_{os.path.basename(registered_filename)}")
        cv.imwrite(unregistered_filename, cv.imread(registered_filename, cv.IMREAD_GRAYSCALE))  # Save the original IR image
        return unregistered_filename


#def combined_registration(vis_image_path, ir_image_path, output_path, image_type):
    try:
        print("Starting combined registration...")

        # 1. Initial Alignment with SIFT (use your registration_broad function)
        initial_registered_path = registration_broad2(vis_image_path, ir_image_path, output_path, image_type)
        if initial_registered_path is ir_image_path:  # Handle failure of SIFT registration
            raise ValueError("SIFT2 registration failed")

        # 2. Refinement with Mutual Information
        return mutual_information_registration2(vis_image_path, initial_registered_path, output_path, image_type)

    except Exception as e:
        print("Combined registration failed:", e)

        # Save the input image to the output directory with a prefix
        unregistered_filename = os.path.join(output_path, f"unregistered_{image_type}_{os.path.basename(ir_image_path)}")
        cv.imwrite(unregistered_filename, cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE))  # Save the original IR image
        return unregistered_filename
    
def orb_registration(vis_image_path, ir_image_path, output_path, image_type):
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
            return mutual_information_registration2(vis_image_path, ir_image_path, output_path, image_type)

        except Exception as e:
            print("ORB-based registration failed:", e)
            # Save the input image to the output directory with a prefix
            unregistered_filename = os.path.join(output_path, f"unregistered_{image_type}_{os.path.basename(ir_image_path)}")
            cv.imwrite(unregistered_filename, cv.imread(ir_image_path, cv.IMREAD_GRAYSCALE))  # Save the original IR image
            return unregistered_filename