import os
import pandas as pd
import cv2 as cv
import argparse
import numpy as np
import time
from datetime import datetime
from matplotlib import pyplot as plt

# Toma la ruta de la carpeta donde están las imágenes y realizar el análisis de convexidad
# Guarda los resultados en un DataFrame, en las carpetas del dataset, como "no_labels"
# y en un archivo CSV con los tiempos de procesamiento y coordenadas de los ratones detectados.
# El nombre del archivo incluye el parámetro de umbral utilizado.

ruta_carpeta = r'/home/ratoncillo/photoneu/labeled-data-ordered'
ruta_carpeta_2 =r'/home/ratoncillo/photoneu/MovAI.v1.tensorflow'

kernel_3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
kernel_5 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
RESIZE_FACTOR = 2.0
MAJOR_DEFECT_THRESHOLD = 12.0 / RESIZE_FACTOR #6.0 5.0 12.0
ELLIPSE_RATIO_THRESHOLD = 5 # 2.0
MIN_AREA = 750# 500#702 # Menos de esto, no es un ratón.
MAX_AREA = 2800#2462#2.5 * MIN_AREA # Más de esto, no es un ratón.
BW_THRES = 12 # min B/W value for threshold
NUM_MICE = 3
RESIZE_FACTOR_X = 1 # Se autocompleta cuando lee la iagen y hace resize a new_size
RESIZE_FACTOR_Y = 1

# img size original: 640 x 480
normal_size = (480, 640)
new_size = (240, 320)

# TriMouse
x_crop_min = int(normal_size[1]/19) #50
x_crop_max = int(normal_size[1]/25) # 30
y_crop_min = int(normal_size[0]/20) # 30
y_crop_max = int(normal_size[0]/20) # 16 -- 40

# 3mice dataset
#x_crop_min = int(normal_size[1]/5)#int(normal_size[1]/10) #50
#x_crop_max = int(normal_size[1]/5)#int(normal_size[1]/20) # 30
#y_crop_min = int(normal_size[0]/5)#int(normal_size[0]/16) # 30
#y_crop_max = int(normal_size[0]/5) #int(normal_size[0]/20) # 16 -- 40

SEGMENT_COLORS = [(0,255,0),(0,255,255),(255,255,0),(255,0,255)]

##############################################

def init():
    # Lista para almacenar los nombres de los archivos
    nombres_imagenes = []

    # Recorrer todos los archivos en la carpeta
    for archivo in os.listdir(ruta_carpeta):
        # Comprobar si el archivo es una imagen (puedes agregar más extensiones si es necesario)
        if archivo.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            archivo = "/"+archivo
            nombres_imagenes.append(archivo)

    # Crear un DataFrame de Pandas con los nombres de las imágenes
    df = pd.DataFrame(nombres_imagenes, columns=['img_path'])

    return df

def read_image(i, ruta_carpeta, img_name):  
#    img_base = os.path.basename(img_name)
    img_base = os.path.splitext(img_name)[0]
    img_in = ruta_carpeta + img_name
    e1 = cv.getTickCount()
    frame = cv.imread(img_in)
    h,w,_ = frame.shape
    frame_o = frame[x_crop_min:(h-x_crop_max), y_crop_min:(w-y_crop_max)]
    resize_x = w / new_size[1]
    resize_y = h / new_size[0]
    frame = cv.resize(frame_o, (int(normal_size[1]/RESIZE_FACTOR), int(normal_size[0]/RESIZE_FACTOR))) # frame.shape/2
#     print(frame.shape) # 240 x 320
    e2 = cv.getTickCount()
    t_init_resize = (e2 - e1)/cv.getTickFrequency()
    return t_init_resize, frame, resize_x, resize_y

def clean_image(frame, thres):
     e1 = cv.getTickCount()
     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
     frame_norm = cv.normalize(frame_gray, None, alpha = 0, beta = 255, norm_type=cv.NORM_MINMAX)
     e2 = cv.getTickCount()
     t_gray_norm = (e2-e1)/cv.getTickFrequency()
     frame_blur = cv.medianBlur(frame_norm, 5)
     e3 = cv.getTickCount()
     t_blur = (e3-e2)/cv.getTickFrequency()
     frame_blur = cv.equalizeHist(frame_blur)
     e4 = cv.getTickCount()
     t_eq = (e4-e3)/cv.getTickFrequency()
     ret, frame_threshold = cv.threshold( frame_blur, thres, 255, cv.THRESH_BINARY_INV ) #+ cv.THRESH_OTSU )
#     frame_threshold = cv.adaptiveThreshold(frame_blur, 255, 
#                                               cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                               cv.THRESH_BINARY_INV, 11, 2)
     e5 = cv.getTickCount()
     t_thresh = (e5 - e4)/cv.getTickFrequency()
     t_clean = {
         "t_gray_norm": t_gray_norm
         ,"t_blur": t_blur
         ,"t_eq":t_eq
         ,"t_thresh":t_thresh
     }
     return t_clean, frame_threshold     

def erode_image(frame):
     e1 = cv.getTickCount()
     frame_erode = cv.morphologyEx(frame, cv.MORPH_DILATE, kernel_3, iterations = 4)
     e2 = cv.getTickCount()
     t_dilate = (e2 - e1)/cv.getTickFrequency()  
     #TODO. Include mice tail detection
     opening = cv.morphologyEx(frame_erode, cv.MORPH_OPEN, kernel_3, iterations = 6)
     e3 = cv.getTickCount()
     t_open = (e3 - e2)/cv.getTickFrequency()

     t_erode = {
        "t_dilate":t_dilate
        ,"t_open":t_open
     }
     return t_erode, opening    

def detect_blobs(image, ruta_imagen, resize_x = 1, resize_y = 1):
#    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    e1 = cv.getTickCount()
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    e2 = cv.getTickCount()
    t_contours = (e2 - e1 )/cv.getTickFrequency()
    blobs = []
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        orig_x, orig_y, width, height = cv.boundingRect(contour)
        roi_image = image[orig_y:orig_y+height,orig_x:orig_x+width]
        perimeter = cv.arcLength(contour, True)
        blobs.append({
            "id" : i
            , "img_path": ruta_imagen
            , "resize_x": resize_x
            , "resize_y": resize_y
            , "contour" : contour
            , "origin" : (orig_x, orig_y)
            , "size" : (width, height)
            , "roi_image" : roi_image
            , "area" : area
            , "perimeter" : perimeter
            , "type": None
            , "ellipses": []
            ,"ellipse_area": []
            ,"hull": None
            ,"split_contours": []
        })
    e3 = cv.getTickCount()
    t_blobs_loop = (e3 - e2)/cv.getTickFrequency()
    t_blobs = {
        "t_contours": t_contours
        ,"t_blobs_loop": t_blobs_loop
    }
    return t_blobs, blobs
 
# split contour using convexity defects and a minimum of mice to detect
def split_blob_contours(blob, D, min_n_mice):
    D_min = 10
    D_max = 10e6
    min_n_mice = np.floor(2*(blob["area"] + 0.1 * blob["area"]) / (MIN_AREA + MAX_AREA))
    min_n_mice = min(min_n_mice, NUM_MICE) # como máximo es NUM_MICE
    min_n_mice = max(min_n_mice, 2) # como mínimo es 2
#    print("expected N mice:", min_n_mice)    
    D_change_factor = 0.1
    max_n_points = (min_n_mice - 1) * 2 
    min_n_points = 2 
    contour = blob["contour"]
    contour = contour.squeeze()  # Elimina dimensiones extra si es necesario
    contour.tolist()
    blob["hull"] = cv.convexHull(contour)

    hull_idx = cv.convexHull(contour, returnPoints=False)
    defects = cv.convexityDefects(contour, hull_idx) 
    #defects are: [start_point, end_point, far_index, distance]
    # - start_point / end_point: closest points where convexHull and countour match
    # - far_index: index to the point of the contour with maximum distance to convexHull (between start and end)
    # - distance: between convexHull and contour point far_index.

    scores = np.zeros(len(blob["contour"]))

    intersections = []
    inter_points = []
    split_contours = []
    widths = []
    distances = []
    D_final = []
    n_points = 0
    if defects is not None :
#        print("defects:", defects)
        for i, defect in enumerate(np.squeeze(defects, 1)):#defects.shape[0]):
          s,e,f,dist= defect
          start = tuple(contour[s])
          end = tuple(contour[e])
          far = tuple(contour[f])
          width = np.linalg.norm(np.array(start) - np.array(end))
          scores[f] = dist*width
        n_points_old = 0
        n_iterations = 0
        while True:
            n_points = np.sum(scores >= D)
#            print("n_points:", n_points)    
            if D < D_min or D > D_max: # por si acaso
              print("D out of range, stopping")
              break 
            # vamos incrementando  decrementando D hasta tener el numero de puntos deseado
            # para evitar bucles infinitos, vamos modificando el factor de cambio                
            else:
                if n_points > max_n_points: # too many points
                  D = D * (1 + D_change_factor)
#                  print("Increasing D to:", D)
                elif n_points < min_n_points: # too few points
                  D = D * (1- D_change_factor)
#                  print("Decreasing D to:", D)
                else:   # just right    
                    break
                if n_points == n_points_old:
                    D_change_factor *= 1.1 # increase change factor to escape oscillations
                elif n_points != n_points_old:
                    D_change_factor /= 1.1 # reduce change factor to avoid oscillations
            n_points_old = n_points
            n_iterations += 1
            if n_iterations > 100:
                print("Too many iterations, stopping")
                break
#        print("Final D:", D)
        D_final.append(D)
        widths.append(width)
        distances.append(dist)
        D_change_factor = 0.1

        for i, defect in enumerate(np.squeeze(defects, 1)):#defects.shape[0]):
            s,e,f,dist= defect
            far = tuple(contour[f])
            if (scores[f] ) >= D: #MAJOR_DEFECT_THRESHOLD
               intersections.append(f)
               inter_points.append(far)
    intersections.sort()
    if n_points != len(intersections):
        print("Warning: n_points != len(intersections)")
        n_points = len(intersections)
        
    if (n_points == 0): # No se detectan defectos de convexidad
        print("No convexity defects detected.")
        split_contours = [contour]
    elif n_points == 1:
        index_plus = intersections[0] + len(contour)//2
        index_less = intersections[0] - len(contour)//2
        if intersections[0] < len(contour)//2:
            blob["segments"] = [
                contour[intersections[0]:index_plus]
                , np.vstack([contour[index_plus:],contour[:intersections[0]+1]])
    #            ,contour[midle_index:] + contour[:intersections[0]]
            ]
        else:
            blob["segments"] = [
                np.vstack([contour[intersections[0]:],contour[:index_less]])
                ,contour[index_less:intersections[0]+1]
                ]
        split_contours = [
            blob["segments"][0], blob["segments"][1]
        ]
    elif n_points == 2:
        if intersections[0] > intersections[1]:
            intersections[0], intersections[1] = intersections[1], intersections[0]
        blob["segments"] = [
                contour[intersections[0]:intersections[1]+1]
                , np.vstack([contour[intersections[1]:],contour[:intersections[0]+1]])
            ]
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , np.vstack([contour[intersections[1]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1]
        ]
    elif n_points == 3:
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , contour[intersections[1]:intersections[2]+1]
            , np.vstack([contour[intersections[2]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1], blob["segments"][2]
        ]
    elif n_points == 4:
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , contour[intersections[1]:intersections[2]+1]
            , contour[intersections[2]:intersections[3]+1]
            , np.vstack([contour[intersections[3]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1],blob["segments"][2], blob["segments"][3]
        ]
    elif n_points == 5:
        blob["segments"] = [
            contour[intersections[0]:intersections[1]+1]
            , contour[intersections[1]:intersections[2]+1]
            , contour[intersections[2]:intersections[3]+1]
            , contour[intersections[2]:intersections[4]+1]
            , np.vstack([contour[intersections[4]:],contour[:intersections[0]+1]])
        ]
        split_contours = [
            blob["segments"][0], blob["segments"][1],blob["segments"][2], blob["segments"][3], blob["segments"][4]
        ]
    else :
       for i in range(0,len(contour),n_points):
          split_contour = contour[i:i + n_points]
          if len(split_contour)> 5:
             split_contours.append(split_contour)
    blob["split_contours"] = split_contours
    for c in split_contours:
        if len(c) >= 5:
            e = cv.fitEllipse(c)
            ea = (np.pi * (e[1][0]/2) * (e[1][1]/2  ))
            if MIN_AREA < ea < MAX_AREA and max(e[1][0], e[1][1]) / min(e[1][0],e[1][1]) < ELLIPSE_RATIO_THRESHOLD:
                blob["ellipse_area"].append(ea)
                blob["ellipses"].append(e) 
    blob["intersections"] = intersections    
    blob["inter_points"] = inter_points   
    blob["widths"] = widths
    blob["distances"] = distances
    blob["D_final"] = D_final
    
    return blob["ellipses"]

def process_blob(blob, num_mice = NUM_MICE-1):    
    if blob["area"] < MIN_AREA:
        blob["type"] = 0 # small blob
    elif MIN_AREA <= blob["area"] <= MAX_AREA:
        blob["type"] = 1 # medium blob
        e = cv.fitEllipse(blob["contour"])
        ea = (np.pi * (e[1][0]/2) * (e[1][1]/2  ))
        if MIN_AREA < ea < MAX_AREA and max(e[1][0], e[1][1]) / min(e[1][0],e[1][1]) < ELLIPSE_RATIO_THRESHOLD:
            blob["ellipses"].append(e)
            blob["ellipse_area"].append(ea)
#        print("blob ellipse:", blob["ellipses"][-1])
    elif blob["area"] > MAX_AREA:
        blob["type"] = 2 # large blob
        split_blob_contours(blob, 20000, num_mice) # Esto habría que ajustarlo al número de ratones que ya se han detectado


def write_results(image, blobs, thres, write_image_flag=True):
    SEGMENT_COLORS = [(0,255,0),(0,255,255),(255,255,0),(255,0,255)]
    output = image.copy()
    for blob in blobs:
        color = (0, 255, 0) if blob["type"] == 0 else (0, 255, 255) if blob["type"] == 1 else (255, 0, 0)
        cv.drawContours(output, [blob["contour"]], -1, color, 1)
        x, y = blob["origin"]
        w, h = blob["size"]
        cv.rectangle(output, (x, y), (x + w, y + h), color, 1)
        cv.putText(output, f'ID:{blob["id"]} A:{int(blob["area"])}', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        if blob["hull"] is not None:
            cv.drawContours(output, [blob["hull"]], -1, (255, 0, 255), 1)
#            cv.putText(output,str(blob["area"]),blob["origin"],, cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),1)# Add character description
        
        for i, ellipse in enumerate(blob["ellipses"]):
#            if ellipse is not None:
                cv.ellipse(output, ellipse, (255,30,25), 1)
                cv.circle(output,(int(ellipse[0][0]),int(ellipse[0][1])), 3, (255,255,25))
                cv.putText(output,str(int(ellipse[0][0]))+","+str(int(ellipse[0][1])),(int(ellipse[0][0]-40),int(ellipse[0][1]+20)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),1)# Add character description
                cv.putText(output,str(int(blob["ellipse_area"][i])),(int(ellipse[0][0]+10),int(ellipse[0][1]-5)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),1)# Add character description


        
        if blob["split_contours"]:
            for n, split_contour in enumerate(blob["split_contours"]):
                cv.polylines(output, [split_contour], False, SEGMENT_COLORS[n%4], 2)   

        if "inter_points" in blob:
            for point in blob["inter_points"]:
                cv.circle(output, point, 3, (0, 0, 255))
    if write_image_flag and len(blobs) > 0:
        img_out = ruta_carpeta + "/no_labels_rev1/"+ str(thres) +"/"
        try:
            os.mkdir(img_out)
        except Exception as e:
            make__dir = "error"
        img_out += blob["img_path"].replace("/","")  
        print(img_out)
        cv.imwrite(img_out,output)

# Construccion de un DF con todas las filas de un mismo img_path fusionadas
# y las columnas mus_i_x, mus_i_y, mus_i_type renumeradas consecutivamente
def merge_blob_rows(df_blob):
    # Detectar todas las columnas mus_i_x, mus_i_y, mus_i_type
    mus_cols = [c for c in df_blob.columns if c.startswith("mus_")]
    
    # Grupo por img_path
    grouped = df_blob.groupby("img_path")
    merged_rows = []
    for img_path, group in grouped:
        new_row = {"img_path": img_path.lstrip('/')}
        i_new = 1  # índice acumulativo para nueva numeración
        # recorrer cada fila del grupo
        for _, row in group.iterrows():
            new_row["resize_x"] = row["resize_x"]
            new_row["resize_y"] = row["resize_y"]
            # buscar todas las parejas mus_i_x/y/type válidas
            for i in range(1, 10):  # límite alto por seguridad
                x_col, y_col, t_col, s_col = f"mus_{i}_x", f"mus_{i}_y", f"mus_{i}_type", f"mus_{i}_area"
                if s_col not in row or pd.isna(row[s_col]):
                    continue
                # copiar datos a nuevas columnas consecutivas
                new_row[f"mus_{i_new}_x"] = row[x_col]
                new_row[f"mus_{i_new}_y"] = row[y_col]
                new_row[f"mus_{i_new}_type"] = row[t_col]
                new_row[f"mus_{i_new}_area"] = row[s_col]
                i_new += 1
                
        merged_rows.append(new_row)
    
    # reconstruir dataframe fusionado
    df_merged = pd.DataFrame(merged_rows)
    return df_merged

### MAIN ################################

scale_factor_x = RESIZE_FACTOR_X
scale_factor_y = RESIZE_FACTOR_Y
bw_thre = 12 # valor optimo

print(f"Processing BW_THRE={bw_thre}")
total_blobs_local = []
df = init()
latencies = []
for i, img in enumerate(df['img_path']):
    t_read_image, frame, resize_x, resize_y = read_image(i, ruta_carpeta, img)
    t_clean, frame_clean = clean_image(frame, bw_thre)
    t_erode, frame_clean = erode_image(frame_clean)
    t_detect, blobs = detect_blobs(frame_clean, img, resize_x, resize_y)
    scale_factor_x = resize_x
    scale_factor_y = resize_y

    total_blobs_local.append(blobs) # process all blobs in image
    e1 = cv.getTickCount()
    for b in blobs:
#     print("blob area = %2d" % b["area"])
     if b["area"] < NUM_MICE * MAX_AREA:
        # asignamos el numero de ratones según la columna df_moveai a la que pertenece el blob
        #buscamos la columna de df_moveai que tiene el mismo img_path que el blob
#        num_mice = df_moveai.loc[df_moveai["filename"] == b["img_path"], "num_mice"].values[0]
        num_mice = NUM_MICE - 1
        process_blob(b, num_mice)
    e2 = cv.getTickCount()
    t_blobs = (e2 - e1)/cv.getTickFrequency()
    latencies.append({"t_read_image":t_read_image
                     , "t_clean":t_clean
                     , "t_erode":t_erode
                     , "t_detect":t_detect
                     , "t_blobs":t_blobs})
    #write_results(frame, blobs, bw_thre, write_image_flag=True)
fn = ruta_carpeta + "/no_labels_rev1/"+ str(BW_THRES) +"/latencies.csv"
#latencies = [d for sublist in latencies for d in sublist]
#cols = ["t_read_image", "t_gray_norm", "t_blur", "t_eq", "t_thresh", "t_dilate", "t_open", "t_contours", "t_blobs_loop", "t_blobs" ]
#df_times = pd.DataFrame(latencies, columns = cols)
df_times = pd.json_normalize(latencies)
df_times.head()
df_times.to_csv(fn)


total_blobs_flat = [d for sublist in total_blobs_local for d in sublist]
df_blobs_local = pd.DataFrame(total_blobs_flat)
max_n = df_blobs_local["ellipses"].apply(len).max()
for i in range(max_n):
    df_blobs_local[f"mus_{i+1}_x"] = df_blobs_local["ellipses"].apply(
        lambda lst: lst[i][0][0] if i < len(lst) else None
    )
    df_blobs_local[f"mus_{i+1}_y"] = df_blobs_local["ellipses"].apply(
        lambda lst: lst[i][0][1] if i < len(lst) else None
    )
    df_blobs_local[f"mus_{i+1}_type"] = df_blobs_local["type"]
    df_blobs_local[f"mus_{i+1}_area"] = df_blobs_local["ellipse_area"].apply(
        lambda lst: lst[i] if i < len(lst) else None
    )   
df_merged = merge_blob_rows(df_blobs_local)
# df_merged = df_blobs

for i in range(1, 10):
    x_col = f"mus_{i}_x"
    y_col = f"mus_{i}_y"
    area_col = f"mus_{i}_area"
    if x_col in df_merged.columns:
        df_merged[x_col] = df_merged[x_col] * scale_factor_x
    if y_col in df_merged.columns:
        df_merged[y_col] = df_merged[y_col] * scale_factor_y
    if area_col in df_merged.columns:
        df_merged[area_col] = df_merged[area_col] * (scale_factor_x * scale_factor_y)

res_file = ruta_carpeta + "/no_labels_rev1/"+ str(BW_THRES) +"/detected.csv"
df_merged.to_csv(res_file)


#########################################

def detectMice( frame, high_V, t_init_resize ): 
     SEGMENT_COLORS = [(0,255,0),(0,255,255),(255,255,0),(255,0,255)]

     frame_defects = frame.copy()
     e2 = cv.getTickCount()
     frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
     frame_norm = cv.normalize(frame_gray, None, alpha = 0, beta = 255, norm_type=cv.NORM_MINMAX)
     e3 = cv.getTickCount()
     frame_blur = cv.medianBlur(frame_norm, 5)
     e4 = cv.getTickCount()
     ret, frame_threshold = cv.threshold( frame_blur, high_V , 255, cv.THRESH_BINARY_INV ) #+ cv.THRESH_OTSU )
#     frame_threshold = cv.normalize(frame_threshold, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U) #TODO: Esto hace falta?
     e5 = cv.getTickCount()
     frame_erode = cv.morphologyEx(frame_threshold, cv.MORPH_DILATE, kernel_3, iterations = 4)  
#     frame_erode = cv.morphologyEx(frame_threshold, cv.MORPH_ERODE, kernel_3, iterations = 5)  
     e6 = cv.getTickCount()
     opening = cv.morphologyEx(frame_erode, cv.MORPH_OPEN, kernel_3, iterations = 6)
     e7 = cv.getTickCount()
    
     blobs = detect_blobs( opening )
     nc = len(blobs)
#     print("Found %2d blob(s)." % len(blobs))
     if len(blobs) > 0: 
        for blob in blobs:
            blob["areas"] = []
            blob["obb"] = []
            blob["mice"] = []
            e = process_blob(blob)
            for n, split_contour in enumerate(blob["split_contours"]):
#                area = cv.contourArea( split_contour )
#                blob["areas"].append( area )
                rect = cv.minAreaRect(split_contour) # center, size, angle
                blob["obb"].append( rect )
                area = rect[1][0] * rect[1][1]
                blob["areas"].append( area )
                box = cv.boxPoints(rect)
                box = np.intp(box)       
                cv.polylines(frame_defects, [split_contour], False, SEGMENT_COLORS[n%4], 2)   
#            cv.drawContours(frame_defects, [blob["hull"]], 0, (0,255,0), 2)      
            cv.drawContours(frame_defects, [blob["contour"]], 0, (155,100,0), 2)
            for n, p in enumerate(blob["inter_points"]):
                cv.circle(frame_defects, p, 3, (0,0,255))
            for n, e in enumerate(blob["ellipses"]):
                if e == None:
                    continue
                area = blob["areas"][n]
                print(area)
                if( area > MIN_AREA) & (area < MAX_AREA ) & (e[0][0] > 0) & (e[0][1] > 0):
                    cv.ellipse(frame, e, (255,30,25), 1)
                    cv.circle(frame,(int(e[0][0]),int(e[0][1])), 3, (255,255,25))
    #                cv.putText(frame,"mice",(int(e[0][0]-40),int(e[0][1]-10)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),1)# Add character description
                    cv.putText(frame,str(int(e[0][0]))+","+str(int(e[0][1])),(int(e[0][0]-40),int(e[0][1]+20)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),1)# Add character description
                    cv.putText(frame,str(round(area)),(int(e[0][0]+10),int(e[0][1]-5)), cv.FONT_HERSHEY_SIMPLEX, 0.6,(100,255,20),1)# Add character description
    #                cv.drawContours(frame,[box],0,(255,0,25),2)
                    blob["mice"].append(e)
     e8 = cv.getTickCount()
     cv.imshow("opening", opening)
     cv.imshow("Detection of defects", frame_defects)


     t_gray_norm = (e3 - e2)/cv.getTickFrequency()
     t_blur = (e4 - e3)/cv.getTickFrequency()
     t_thres = (e5 - e4)/cv.getTickFrequency()
     t_erosion = (e6 - e5)/cv.getTickFrequency()
     t_opening = (e7 - e6)/cv.getTickFrequency()
     t_blob = (e8 - e7)/cv.getTickFrequency()

     latencies = [t_init_resize, t_gray_norm,t_blur,t_thres,t_erosion,t_opening, t_blob]
     return latencies, blobs, frame, frame_defects    

# Función para leer los nombres de las imágenes de una carpeta
def leer_imagenes_de_carpeta(carpeta):
    # Lista para almacenar los nombres de los archivos
    nombres_imagenes = []
    
    # Recorrer todos los archivos en la carpeta
    for archivo in os.listdir(carpeta):
        # Comprobar si el archivo es una imagen (puedes agregar más extensiones si es necesario)
        if archivo.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')):
            archivo = "/"+archivo
            nombres_imagenes.append(archivo)
    
    # Crear un DataFrame de Pandas con los nombres de las imágenes
    df = pd.DataFrame(nombres_imagenes, columns=['img_path'])
    
    return df

def analyze_image_from_path(i, ruta_carpeta, img_name):  
#    img_base = os.path.basename(img_name)
    img_base = os.path.splitext(img_name)[0]
    img_in = ruta_carpeta + img_name
    e1 = cv.getTickCount()
    frame = cv.imread(img_in)
    frame_o = frame[x_crop_min:(normal_size[0]-x_crop_max), y_crop_min:(normal_size[1]-y_crop_max)]
    frame = cv.resize(frame_o, (int(normal_size[1]/RESIZE_FACTOR), int(normal_size[0]/RESIZE_FACTOR))) # frame.shape/2
#     print(frame.shape) # 240 x 320
    e2 = cv.getTickCount()
    t_init_resize = (e2 - e1)/cv.getTickFrequency()

    _,_, final_frame,defects_frame = analyze_image(i, frame = frame, t_init_resize = t_init_resize)
    write_image(ruta_carpeta, img_name, final_frame)
    write_image(ruta_carpeta, img_base + "_c" + ".png", defects_frame)


def analyze_image( i, frame, t_init_resize ):  
    t, b, final_frame, convex_frame = detectMice(frame, thres, t_init_resize)
    k = 0
    nip = 0
    for j, blob in enumerate(b):
        kk = 0
        for jj, m in enumerate(blob["mice"]):
            cx, cy = m[0]
            if k < 3: 
                mus_x[k,i] = int(cx)
                mus_y[k,i] = int(cy)
                a = int(blob["areas"][kk])
                area[k,i] = a
                k += 1
                kk += 1
        nip += len(blob["intersections"])
    times.append(t)
    nb.append(len(b))
    num_inter_points.append(nip)
    if nip != 0:
        if_overlaps.append(1)
    else:
        if_overlaps.append(0)
    num_mice_per_blob = 0
    for n, blob in enumerate(b):
        num_mice_per_blob += len(blob["mice"])
    nm.append(num_mice_per_blob)
    return t, b, final_frame, convex_frame
    
def write_image(ruta_carpeta, img_name, final_frame):
    img_out = ruta_carpeta + "/no_labels/"+ str(thres) +"/"
    try:
        os.mkdir(img_out)
    except Exception as e:
        print(f"{e}")
    img_out += img_name  
    print(img_out)
    cv.imwrite(img_out, final_frame)
    
def analyze_video(video_path):
    cap = cv.VideoCapture(video_path)
#    frame_width = int(cap.get(3))
#    frame_height = int(cap.get(4))
    frame_width = int(normal_size[0]/RESIZE_FACTOR)
    frame_height = int(normal_size[1]/RESIZE_FACTOR)
    convex_out = cv.VideoWriter('no_labels_convex_5.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_height,frame_width)) 
    video_out = cv.VideoWriter('no_labels_5.avi',cv.VideoWriter_fourcc('M','J','P','G'), 10, (frame_height,frame_width))
    i = 0

    while cap.isOpened():
        e1 = cv.getTickCount()
        ret, frame = cap.read()
        if frame is None:
            print("frame ERROR")
            break
        frame = cv.resize(frame, (frame_height, frame_width)) # frame.shape/2
        e2 = cv.getTickCount()
        t_init_resize = (e2 - e1)/cv.getTickFrequency()
        _,_,frame_out, frame_convex = analyze_image(i, frame, t_init_resize)
        video_out.write(frame_out)
        convex_out.write(frame_convex)
        cv.imshow("video", frame_out)
        i+=1
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    print("n_frames = " + str(i))
    cap.release()
        
def updateDataFrame():
    df_times = pd.DataFrame(times, columns=['t_init_resize', 't_gray_norm','t_blur','t_thres','t_erosion','t_opening', 't_blobs'])
    df_tmp = df.assign(num_mice = nm)
    df_tmp = df_tmp.assign(num_blobs = nb)
    df_tmp = df_tmp.assign(num_inter_points = num_inter_points)
    df_tmp = df_tmp.assign(overlaps = if_overlaps)
    df_tmp = df_tmp.assign(mus_1_x = mus_x[0])
    df_tmp = df_tmp.assign(mus_2_x = mus_x[1])
    df_tmp = df_tmp.assign(mus_3_x = mus_x[2])
    df_tmp = df_tmp.assign(mus_1_y = mus_y[0])
    df_tmp = df_tmp.assign(mus_2_y = mus_y[1])
    df_tmp = df_tmp.assign(mus_3_y = mus_y[2])
    df_tmp = df_tmp.assign(mus_1_area = area[0])
    df_tmp = df_tmp.assign(mus_2_area = area[1])
    df_tmp = df_tmp.assign(mus_3_area = area[2])
#    print(df_times)
    print(df_tmp[df_tmp["num_mice"] >= 3])
    print("n_total_mice = " + str(sum( df_tmp["num_mice"] )))
    print("n_files_3_mices = " + str(df_tmp[df_tmp["num_mice"] == 3].shape[0]))
    fn = "../logs/"
    fn += datetime.now().strftime("%Y_%m_%d_%I_%M_%S") 
    fn += "_no_labels_test_" + str(thres) + ".csv"
    
    #df_times.to_csv('../logs/no_labels_test_times.csv')
    print(fn)
    df_tmp.to_csv(fn)

