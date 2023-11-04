import cv2
import numpy as np

def sparse_optical_flow(prev_gray, gray, prev_pts):
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
    good_old = prev_pts[status == 1]
    good_new = new_pts[status == 1]
    return good_old, good_new

def dense_optical_flow(prev_gray, gray):
    return cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

def process_video(input_video):
    cap = cv2.VideoCapture(input_video)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape[:2]
    y, x = np.mgrid[10:h:20, 10:w:20].reshape(2, -1).astype(np.float32)
    prev_pts = np.vstack((x, y)).T
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (w, h))
    
    arrow_start = (50, 50)
    outlier_arrow_start = (arrow_start[0] + 50, arrow_start[1])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        good_old, good_new = sparse_optical_flow(prev_gray, gray, prev_pts)
        flow = dense_optical_flow(prev_gray, gray)

        motion = good_new - good_old
        avg_motion = np.mean(motion, axis=0)
        avg_motion = np.squeeze(avg_motion)

        mean_intensity = np.mean(gray)
        dark_indices = gray < mean_intensity
        motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_magnitude = np.mean(motion_magnitude[dark_indices])
        std_magnitude = np.std(motion_magnitude[dark_indices])
        outliers = (np.abs(motion_magnitude - avg_magnitude) > 2 * std_magnitude) & dark_indices
        
        frame_out = frame.copy()
        
        # Find outliers of outliers
        outlier_motion = flow[outliers]
        if len(outlier_motion) > 0:
            avg_outlier_motion = np.mean(outlier_motion, axis=0)
            avg_outlier_motion = np.squeeze(avg_outlier_motion)
            avg_outlier_magnitude = np.sqrt(avg_outlier_motion[0]**2 + avg_outlier_motion[1]**2)
            std_outlier_magnitude = np.std(np.sqrt(outlier_motion[:, 0]**2 + outlier_motion[:, 1]**2))
            
            outlier_of_outliers = (np.abs(motion_magnitude[outliers] - avg_outlier_magnitude) > 2 * std_outlier_magnitude)
            final_outliers = np.zeros_like(outliers)
            final_outliers[outliers] = outlier_of_outliers
            
            frame_out[final_outliers] = [0, 0, 255]  # Outliers of outliers in red

        # Draw arrows
        if np.any(avg_motion):
            arrow_end = (int(arrow_start[0] + avg_motion[0]*10), int(arrow_start[1] + avg_motion[1]*10))
            cv2.arrowedLine(frame_out, arrow_start, arrow_end, (0, 255, 0), 2, tipLength=0.3)

        if np.any(avg_outlier_motion):
            outlier_arrow_end = (int(outlier_arrow_start[0] + avg_outlier_motion[0]*10),
                                  int(outlier_arrow_start[1] + avg_outlier_motion[1]*10))
            cv2.arrowedLine(frame_out, outlier_arrow_start, outlier_arrow_end, (0, 255, 255), 2, tipLength=0.3)
        
        out.write(frame_out)
        cv2.imshow("Outliers of Outliers", frame_out)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        prev_gray = gray.copy()
        prev_pts = good_new.reshape(-1, 1, 2)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

process_video('path_to_your_video.mp4')
