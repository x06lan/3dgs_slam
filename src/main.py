import torch
import cv2
import time
import numpy as np

# import gaussian_cuda

from parser.dataset import ColmapDataset
from tracker.visual_odometry import VisualOdometry
from tracker.feature_tracker_configs import FeatureTrackerConfigs
from tracker.feature_tracker import feature_tracker_factory, FeatureTrackerTypes
from viewer.mplot_thread import Mplot2d, Mplot3d

if __name__ == "__main__":

    # dataset = ColampDataset("./dataset/nerfstudio/person", 2)
    dataset = ColmapDataset("./dataset/nerfstudio/redwoods2", 2)
    images, images_info, camera = dataset.dump()

    num_features = 100  # how many features do you want to detect and track?
    tracker_config = FeatureTrackerConfigs.LK_FAST
    tracker_config['num_features'] = num_features
    feature_tracker = feature_tracker_factory(**tracker_config)

    # matched_points_plt = Mplot2d(
    #     xlabel='img id', ylabel='# matches', title='# matches')

    vo = VisualOdometry(camera,  feature_tracker)

    draw_scale = 1
    traj_img_size = 800
    half_traj_img_size = int(0.5*traj_img_size)
    traj_img = np.zeros(
        (traj_img_size, traj_img_size, 3), dtype=np.uint8)
    # traj_img = traj_img.numpy()

    for img_id in range(len(images)):
        image = images[img_id]
        image_numpy = image.numpy()

        image_numpy = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
        if (image is not None):
            vo.track(image_numpy, img_id)
        else:
            break

        if (img_id > 2):
            x, y, z = vo.traj3d_est[-1]
            x_true, y_true, z_true = vo.traj3d_gt[-1]
            # print(image[0][0])
            # print(image[0][0]-vo.draw_img[0][0])

            draw_x, draw_y = int(
                draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
            true_x, true_y = int(
                draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
            # estimated from green to blue
            cv2.circle(traj_img, (draw_x, draw_y), 1,
                       (img_id*255/4540, 255-img_id*255/4540, 0), 1)
            cv2.circle(traj_img, (true_x, true_y), 1,
                       (0, 0, 255), 1)  # groundtruth in red
            # write text on traj_img
            cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
            text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
            cv2.putText(traj_img, text, (20, 40),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
            # show
            cv2.imshow('Trajectory', traj_img)

            cv2.imshow('vo', vo.draw_img)
            cv2.imshow('camera', image_numpy)

        cv2.waitKey(1)
        # time.sleep(0.1)

    # err_plt.quit()
    cv2.destroyAllWindows()
