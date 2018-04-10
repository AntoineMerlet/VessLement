int main() 
{
	try
	{	
		std::string dataset_path = "V:/EIID/projects/AIA-Retinal-Vessel-Segmentation/dataset/";
		std::vector <cv::Mat> images = aia::getImagesInFolder(dataset_path + "images", ".tif");
		std::vector <cv::Mat> truths = aia::getImagesInFolder(dataset_path + "groundtruth", ".tif", true);
		std::vector <cv::Mat> masks  = aia::getImagesInFolder(dataset_path + "mask", ".tif", true);

		// dummy segmentation: thresholding / binarization
		for(auto & im : images)
		{
			// first convert to gray
			cv::cvtColor(im, im, CV_BGR2GRAY);

			// then invert so that vessels are bright
			im = 255 - im;

			// apply thresholding
			cv::threshold(im, im, 160, 255, CV_THRESH_BINARY);
		}

		std::vector <cv::Mat> visual_results;
		double ACC = aia::accuracy(images, truths, masks, &visual_results);
		printf("Accuracy (dummy segmentation) = %.2f%%\n", ACC*100);
		//for(auto & v : visual_results)
			//aia::imshow("Visual result", v);


		// ideal segmentation: use the groundtruth (we should get 100% accuracy)
		ACC = aia::accuracy(truths, truths, masks);
		printf("Accuracy (ideal segmentation) = %.2f%%\n", ACC*100);

		return 1;
	}
	catch (aia::error &ex)
	{
		std::cout << "EXCEPTION thrown by " << ex.getSource() << "source :\n\t|=> " << ex.what() << std::endl;
	}
	catch (ucas::Error &ex)
	{
		std::cout << "EXCEPTION thrown by unknown source :\n\t|=> " << ex.what() << std::endl;
	}
}