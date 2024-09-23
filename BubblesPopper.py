import cv2
import numpy as np
import random
from cvzone.SelfiSegmentationModule import SelfiSegmentation

#Open camera
url = 'http://10.82.22.195:4747/video'
vcap = cv2.VideoCapture(0) #Change url to 0 to use PC camera, url to use phone camera (use ur IP address)

# Set custom width and height for the webcam window
frame_width = 1000  # Desired width
frame_height = 700  # Desired height

# Create a named window with a fixed size
cv2.namedWindow('Bubble Popper Game', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Bubble Popper Game', frame_width, frame_height)

seg = SelfiSegmentation()

#Define color range for the object (blue)
lower_color = np.array([100, 150, 0])
upper_color = np.array([140, 255, 255])

#Variables
score = 0
highest_score = 0
elapsed_time = 30
game_over = False
reset = False
button_clicked = False

# Load the bubble image
bubble_img = cv2.imread('img/bubble.png', cv2.IMREAD_UNCHANGED)  
# Load the CPU Logo
cpu_logo = cv2.imread('img/Cpu.jpg', cv2.IMREAD_UNCHANGED)  

#Restart button
button_width, button_height = 180, 50
button_x, button_y = int(frame_width/2 - button_width/2), int(frame_height/2 - button_height/2)

#Generate random bubbles
bubbles = []
bubble_speed = 5  # Speed at which bubbles fall
bubble_creation_rate = 10  # Create a new bubble every X frames
frame_count = 0  # To control bubble creation
for _ in range(5):
    x = random.randint(50, 600) #Position for a bubble
    y = random.randint(50, 400)
    radius= random.randint(20, 50) #Bubble size
    bubbles.append([x, y, radius])

def overlay_image(background, overlay, x, y):
    """Overlays `overlay` image onto `background` at position (x, y)."""
    overlay_h, overlay_w = overlay.shape[:2]
    background_h, background_w = background.shape[:2]

    # Ensure the overlay doesn't go out of bounds (clip the overlay to fit the frame)
    if x < 0:  
        overlay_w += x  
        overlay = overlay[:, -x:]  
        x = 0

    if y < 0:  
        overlay_h += y  
        overlay = overlay[-y:, :]  
        y = 0

    if x + overlay_w > background_w:  
        overlay_w = background_w - x  
        overlay = overlay[:, :overlay_w]

    if y + overlay_h > background_h:  
        overlay_h = background_h - y  
        overlay = overlay[:overlay_h, :]

    # Extract the region of interest (roi) from the background where the overlay will be placed
    if overlay_h > 0 and overlay_w > 0:  # Ensure valid dimensions
        roi = background[y:y+overlay_h, x:x+overlay_w]
    else:
        return background  # Skip overlay if invalid dimensions

    # Check if the overlay image has an alpha channel (transparency)
    if overlay.shape[2] == 4:  # RGBA image (with transparency)
        # Split the overlay image into its color and alpha channels
        overlay_rgb = overlay[:, :, :3]  # Get the first 3 channels (BGR)
        mask = overlay[:, :, 3]  # Get the alpha channel

        if mask is None or mask.size == 0:
            print("Warning: Empty mask encountered, skipping overlay.")
            return background

        # Create inverse mask
        mask_inv = cv2.bitwise_not(mask)

        if mask_inv is None or mask_inv.size == 0:
            print("Warning: Empty inverse mask encountered, skipping overlay.")
            return background

        # Convert the mask to 3 channels for compatibility with ROI
        mask_inv_rgb = cv2.merge([mask_inv, mask_inv, mask_inv])

        if mask_inv_rgb is None or mask_inv_rgb.size == 0:
            print("Warning: Empty mask_inv_rgb encountered, skipping overlay.")
            return background

        # Black-out the area of the overlay in ROI
        background_bg = cv2.bitwise_and(roi, mask_inv_rgb)

        # Take only the overlay region
        overlay_fg = cv2.bitwise_and(overlay_rgb, overlay_rgb, mask=mask)

        # Add the overlay and the background
        combined = cv2.add(background_bg, overlay_fg)
        background[y:y+overlay_h, x:x+overlay_w] = combined
    else:
        # Simply overlay if there's no transparency (no alpha channel)
        background[y:y+overlay_h, x:x+overlay_w] = overlay

    return background

def draw_scoreboard(frame, score, highest_score, elapsed_time):
    """Draws the scoreboard in the top-left corner of the frame."""
    # Define the rectangle position and size
    rect_x, rect_y = 10, 10
    rect_width, rect_height = 220, 100
 
    # Define text color
    text_color = (255, 255, 255)  # White

    # Define font type and scale
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    if not game_over:
        # Draw the rectangle
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)  # Black rectangle
        # Draw the score text
        cv2.putText(frame, f'Score: {score}', (rect_x + 10, rect_y + 30), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f'Highest Score: {highest_score}', (rect_x + 10, rect_y + 60), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f'Time: {elapsed_time:.1f}s', (rect_x + 10, rect_y + 90), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    if game_over:
        # Draw the restart button
        cv2.rectangle(frame, (button_x, button_y), (button_x + button_width, button_y + button_height), (255, 0, 0), -1)  # Green button
        cv2.putText(frame, 'Restart', (button_x + 50, button_y + 30), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        # Draw the game over message
        gorect_x, gorect_y = button_x-70, button_y+button_height+10
        gorect_width, gorect_height = 300, 120
        cv2.rectangle(frame, (gorect_x, gorect_y), (gorect_x + gorect_width, gorect_y + gorect_height), (0, 0, 255), -1)  # Red button
        cv2.putText(frame, f"Time is Over!", (gorect_x+55, gorect_y+30), font, font_scale * 1.2, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Score: {score}", (gorect_x+5, gorect_y+65), font, font_scale * 1.2, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.putText(frame, f"Highest Score: {highest_score}", (gorect_x+5, gorect_y+100), font, font_scale * 1.2, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return frame

def handle_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if game_over:
            # Check if click is within the button
            if button_x < x < button_x + button_width and button_y < y < button_y + button_height:
                global button_clicked
                button_clicked = True 
                print(button_clicked)

flag = 0

cv2.setMouseCallback('Bubble Popper Game', handle_click)
while True:
    ret, frame = vcap.read()  # Capture each frame from the webcam
    if not ret:
        break

    # Resize the captured frame to the desired size
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Apply background removal using cvzone
    vid_rmbg = seg.removeBG(frame, cpu_logo, cutThreshold=0.8)

    # Resize and overlay CPU logo
    resized_cpu_logo = cv2.resize(cpu_logo, (50, 50))  # Adjusted logo size for visibility
    vid_rmbg = overlay_image(vid_rmbg, resized_cpu_logo, -20, -130)  # Adjusted position for logo

    # Display the background removed feed
    cv2.imshow('Bubble Popper Game', vid_rmbg)

    # Game logic: frame count for timing and checking if game is over
    frame_count += 1
    if not (frame_count % 30) and not game_over:
        elapsed_time -= 1
        if elapsed_time <= 0:
            game_over = True

    # Convert frame to HSV for object tracking
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the tracked object (based on HSV range)
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Find contours for the tracked object
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours and detect object-bubble collision
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = (int(x), int(y))
        cv2.circle(vid_rmbg, center, int(radius), (0, 255, 0), 2)  # Draw circle around the tracked object

        # Check for collision with bubbles
        for bubble in bubbles[:]:
            bubble_x, bubble_y, bubble_radius = bubble
            distance = np.sqrt((center[0] - bubble_x) ** 2 + (center[1] - bubble_y) ** 2)

            if distance < bubble_radius + radius:  # If object touches the bubble
                bubbles.remove(bubble)  # Pop the bubble
                score += 1
                if score > highest_score:
                    highest_score = score

    if not game_over:
        # Generate new bubbles at random positions at the top of the screen
        if frame_count % bubble_creation_rate == 0:
            x = random.randint(50, frame.shape[1] - 50)  # Horizontal position
            radius = random.randint(20, 50)  # Random radius
            bubbles.append([x, 0, radius])  # Add a new bubble at the top

        # Move bubbles down the screen and remove off-screen bubbles
        for bubble in bubbles[:]:
            bubble[1] += bubble_speed  # Move the bubble downwards
            if bubble[1] - bubble[2] > frame.shape[0]:  # If the bubble is off the screen
                bubbles.remove(bubble)  # Remove the bubble

        # Draw remaining bubbles on top of the frame
        for bubble in bubbles:
            resized_bubble_img = cv2.resize(bubble_img, (bubble[2] * 2, bubble[2] * 2))  # Resize bubble image to match size
            vid_rmbg = overlay_image(vid_rmbg, resized_bubble_img, bubble[0] - bubble[2], bubble[1] - bubble[2])

    # Draw the scoreboard on top of everything
    vid_rmbg = draw_scoreboard(vid_rmbg, score, highest_score, elapsed_time)

    # Show the frame with the bubbles and scoreboard
    cv2.imshow('Bubble Popper Game', vid_rmbg)

    # Handle keypresses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to quit
        break

    # # Capture background
    # elif key == ord('d'):
    #     flag = 1
    #     print("Background Captured")

    # Reset the game when game is over and the reset button is clicked
    elif button_clicked and game_over:
        print("Game reset!")
        score = 0
        elapsed_time = 30
        game_over = False
        button_clicked = False

# Release resources
vcap.release()
cv2.destroyAllWindows()
