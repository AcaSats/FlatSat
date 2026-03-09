import heapq
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_POI(ml_img_ORIG, combined_crater_img_path, img_path, contours): # Get points of interest
    print(combined_crater_img_path)
    combined_crater_img = cv2.imread(combined_crater_img_path)

    # Get original point of interest
    hsv = cv2.cvtColor(ml_img_ORIG, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([11, 100, 100])
    upper_orange = np.array([25, 255, 255])

    POI_mask_red = cv2.inRange(hsv, lower_orange, upper_orange)
               
    # Dilate mask to make it bigger than boundary
    kernel = np.ones((12, 12), np.uint8)
    dilated_mask = cv2.dilate(POI_mask_red, kernel, iterations=1)
    orig_POI_contour, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(orig_POI_contour)

    if (len(orig_POI_contour) < 2): # if not two points of interest, search for largest crater
        # Find point of interest (biggest circled boundary)
        largest_contour = max(contours, key=cv2.contourArea)

        # Dilate contour to make it bigger than boundary
        mask = np.zeros((combined_crater_img.shape[0], combined_crater_img.shape[1]), dtype=np.uint8)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=-1) 
        kernel = np.ones((3, 3), np.uint8)
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        new_contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_largest_contour = max(new_contours, key=cv2.contourArea)

        point_of_interest = ml_img_ORIG.copy()
        cv2.drawContours(combined_crater_img, [new_largest_contour], -1, (0, 0, 255), -1)

    # Draw point of interest on top
    cv2.drawContours(combined_crater_img, orig_POI_contour, -1, (0, 0, 255), -1)
    cv2.imwrite(img_path, combined_crater_img)

def line_of_sight(p1, p2, grid):
    x1, y1 = p1
    x2, y2 = p2

    points = np.linspace((x1, y1), (x2, y2), 100)

    for x, y in points:
        if grid[int(y), int(x)] == 0:
            return False
    return True

def heuristic(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def theta_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0, start))

    parent = {start: start}
    g = {start: 0}

    directions = [
        (-1,0),(1,0),(0,-1),(0,1),
        (-1,-1),(-1,1),(1,-1),(1,1)
    ]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current != parent[current]:
                path.append(current)
                current = parent[current]
            path.append(start)
            return path[::-1]

        x, y = current

        for dx, dy in directions:
            nx, ny = x+dx, y+dy

            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if grid[ny, nx] == 0:
                continue

            neighbor = (nx, ny)

            if parent[current] != current and line_of_sight(parent[current], neighbor, grid):
                # Tvry shortcut via parent
                new_g = g[parent[current]] + heuristic(parent[current], neighbor)
                if neighbor not in g or new_g < g[neighbor]:
                    g[neighbor] = new_g
                    parent[neighbor] = parent[current]
                    f = new_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
            else:
                # Normal update
                new_g = g[current] + heuristic(current, neighbor)
                if neighbor not in g or new_g < g[neighbor]:
                    g[neighbor] = new_g
                    parent[neighbor] = current
                    f = new_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))

    return None

def process_img(img_path):
    img_crater = cv2.imread(img_path)
    if img_crater is None:
        raise ValueError("Could not load the image.")

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img_crater, cv2.COLOR_BGR2HSV)

    # Red mask to find points of interest
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_orange = np.array([11, 100, 100])
    upper_orange = np.array([25, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | \
               cv2.inRange(hsv, lower_red2, upper_red2) | \
               cv2.inRange(hsv, lower_orange, upper_orange) # likely craters

    # Green mask to find obstacles
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green, upper_green) # points of interest
    
    return [mask_red, mask_green]

def plot_path(mask_red, mask_green, file_path, img_path):
    img_crater = cv2.imread(img_path)
    contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find points of interest
    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    start = centers[0]
    goal = centers[1]
    print(start, goal)

    kernel = np.ones((15,15), np.uint8)  # increase size to increase safety distance
    inflated_green = cv2.dilate(mask_green, kernel, iterations=1)

    grid = np.ones(mask_green.shape, dtype=np.uint8)
    grid[inflated_green > 0] = 0
    
    path = theta_star(grid, start, goal)
    if path:
        pts = np.array(path, np.int32).reshape((-1,1,2))
        output = img_crater.copy()

        cv2.polylines(
            output,
            [np.array(path, np.int32).reshape((-1,1,2))],
            False,
            (0,0,255),
            4,
            lineType=cv2.LINE_AA
        )
        
        cv2.imwrite(file_path, output)
        
    else:
        print("No path found")
        
def path_finding(orig_img_path, ml_img_ORIG, crater_img_path, crater_contour): # path to marked up image with craters    
    # print(crater_img_path)
    path_arr = orig_img_path.split('/')
    img_name = path_arr[len(path_arr) - 1].split('.')[0] + '_combined_craters_and_POI.png'
    combined_POI_path = path_arr[:len(orig_img_path.split('/'))-1]
    combined_POI_path = '/'.join(combined_POI_path)
    combined_POI_path += '/' + img_name
    get_POI(ml_img_ORIG, crater_img_path, combined_POI_path, crater_contour)
    
    [mask_red, mask_green] = process_img(combined_POI_path)
    
    img_name = path_arr[len(path_arr) - 1].split('.')[0] + '_path.png'
    file_path_to_path = path_arr[:len(orig_img_path.split('/'))-1]
    file_path_to_path = '/'.join(file_path_to_path)
    file_path_to_path += '/' + img_name
    
    plot_path(mask_red, mask_green, file_path_to_path, combined_POI_path);
    
    
