# -*- coding: utf-8 -*1-
"""
Created on Sat Jul 14 14:23:41 2012

@author: Max Lungarella
"""

import cv2.cv as cv
import cv2
import numpy as np
import os
import time

# -----------------------------------------------------------------------------
''' Histogram equalization
    Flattens the graylevel histogram so that all intensities are as equally
    common as possible
'''
def histeq( im, nbr_bins=256 ):
    im = np.array( im )

    # Get image histogram    
    imhist, bins = np.histogram( im.flatten(), nbr_bins, normed=True )

    # Cumulative distribution function    
    cdf = imhist.cumsum()

    # Normalize in [0,255]
    cdf = 255 * cdf / cdf[-1]

    # Use linear interpolation of cdf to find new pixel values
    im2 = np.interp( im.flatten(), bins[:-1], cdf )
    
    return np.uint8(im2.reshape( im.shape )), cdf    

# -----------------------------------------------------------------------------
''' Marker target
    Saves marker_target.png to file; typically a checkerboard pattern or the like
'''
def marker_target():
    marker_target = cv.CreateImage( (620,620), 8, 3 )
    cv.Set( marker_target, (255,255,255) )
    
    for i in range(300/20):
        cv.Rectangle( marker_target, (0,i*2*20), (20,(i*2+1)*20), (0,0,0), thickness=-1 )
        cv.Rectangle( marker_target, (600,i*2*20), (620,(i*2+1)*20), (0,0,0), thickness=-1 )
        cv.Rectangle( marker_target, (i*2*20,0), ((i*2+1)*20,20), (0,0,0), thickness=-1 )
        cv.Rectangle( marker_target, (i*2*20,600), ((i*2+1)*20,620), (0,0,0), thickness=-1 )
    cv.Rectangle( marker_target, (600,600), (620,620), (0,0,0), thickness=-1 )
    
    cv.ShowImage ( 'marker_target', marker_target )
    cv.SaveImage( 'marker_target.png', marker_target )


# -----------------------------------------------------------------------------
''' Scale
'''
def add_scale( img ):
    h,w = img.shape[:2]
    pix_5mm = 160/2.0
    d5 = 5
    pix_1mm = pix_5mm/5.0
    d1 = 3
    w0 = 10
    h0 = h-10
    font = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 0.8, 0.8, 0, 1, cv.CV_AA)    
    arr = cv.fromarray( img )    

    # Vertical line
    cv.Line( arr, (w0,10), (w0,h-10), color=(255,255,255), thickness=1, lineType=cv.CV_AA )
    # Horizontal line
    cv.Line( arr, (w0,h0), (w-10,h0), color=(255,255,255), thickness=1 )    
    # Vertical axis primary tics
    for i in range( np.int((h-20)/pix_5mm)+1 ):
        hf = np.int(h0-i*pix_5mm)
        cv.Line( arr, (w0,hf), (w0+d5,hf), color=(255,255,255), thickness=1 )
        if i>0:
            cv.PutText( arr, str(i*5), (w0+2*d5,hf+5), font, (255,255,255) )
    # Vertical axis secondary tics
    for i in range( np.int((h-20)/pix_1mm)+1 ):
        hf = np.int(h0-i*pix_1mm)
        cv.Line( arr, (w0,hf), (w0+d1,hf), color=(255,255,255), thickness=1 )
    # Horizontal axis main tics
    for i in range( np.int((w-20)/pix_5mm)+1 ):
        wf = np.int(w0+i*pix_5mm)
        cv.Line( arr, (wf,h0), (wf,h0-d5), color=(255,255,255), thickness=1 ) 
        if i>0:        
            cv.PutText( arr, str(i*5), (wf-d5,h0-2*d5), font, (255,255,255) )
    # Horizontal axis secondary tics
    for i in range( np.int((w-20)/pix_1mm)+1 ):
        wf = np.int(w0+i*pix_1mm)
        cv.Line( arr, (wf,h0), (wf,h0-d1), color=(255,255,255), thickness=1 ) 
     
    cv.PutText( arr, 'in mm', (w-50,20), font, (255,255,255) ) 
    
    return arr

# -----------------------------------------------------------------------------
''' Auto crop
    Automatically cuts out the relevant part of an image; the flag display enables
    or disables the display of all intermediate steps
'''
def pic_cutout( img, img_dst, name='cropped-final', display=False ):
    if display==True:    
        cv2.imshow( 'orig', img ) 

    # Make gray image   
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    if display==True:
        cv2.imshow( 'gray', gray )
        
    # Equalize histogram 
    gray_norm, cdf = histeq( gray, 256 )            
    if display==True:    
        cv2.imshow( 'gray_norm', gray_norm )    
    
    # Apply Otsu's binarization (global thresholding)
    ret, thresh = cv2.threshold( gray_norm, 32, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )    
    if display==True:     
        cv2.imshow( 'thresh', thresh ) 

    # Apply erosion + dilation -> opening
    fg = cv2.erode( thresh, None, iterations=3 )
    bgt = cv2.dilate( thresh, None, iterations=2 )

    #bg = cv2.adaptiveThreshold( bgt, 64, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 1 )    
    ret, bg = cv2.threshold( bgt, 1, 128, cv2.THRESH_BINARY_INV ) 
    
    # Colors: white -> erosion, gray -> background, black -> dilation
    marker = cv2.add( fg, bg )  # marker.dtype = dtype('uint8')
    if display==True:    
        cv2.imshow( 'marker', marker )    

    # Convert to 32-bit integers32
    marker32 = np.int32( marker )   # marker32.dtype = dtype('int32')    

    # Apply watershed algorithm
    cv2.watershed( img, marker32 )
    img_abs = cv2.convertScaleAbs( marker32 )    

    # Final thresholding
    ret, thresh = cv2.threshold( img_abs, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU )
    cv2.bitwise_and( img, img, mask=thresh )
    
    # Returns a Python list object!
    contours0, hierarchy = cv2.findContours( thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    contours = [cv2.approxPolyDP(cnt, 1, True) for cnt in contours0]

    if display==True:    
        cv2.drawContours( img, contours, -1, (0,255,0), 1)
        cv2.imshow( 'contours', img )
    
    # Finds the largest rect bounding box; does not consider rotation of object!
    x=y=w=h = 0
    min_rect = 0
    max_size = float("-inf")
    max_cnt = None
    for cnt in contours:
        size = abs(cv2.contourArea( cnt ))        
        if size>max_size:
            max_size = size
            max_cnt = cnt

        x,y,w,h = cv2.boundingRect( max_cnt )
        min_rect = cv2.minAreaRect( max_cnt )

    # Draws minimum area rectangle
    box = cv2.cv.BoxPoints( min_rect )
    box = np.int0( box )
    if display==True:
        cv2.drawContours( img, [box], 0, (0,0,255), 2 )
        cv2.imshow( 'min area rect', img )

    # Draws bounding rectangle      
    cv2.rectangle( img, (x,y), (x+w,y+h), (0,255,0), 1)
    if display==True:
        cv2.imshow( 'bounding', img )
    print str(name), x, y, w, h, max_size   
    
    # Set region of interest    
    bound = 40
    a = 0    
    if x>bound and y>bound:     
        a = img_dst[y-bound:y+h+bound, x-bound:x+w+bound]
    elif y<=bound:
        a = img_dst[0:y+h+bound, x-bound:x+w+bound]
    elif x<=bound:
        a = img_dst[y-bound:y+h+bound, 0:x+w+bound]
    else:
        a = img_dst[0:y+h+bound, 0:x+w+bound]
    
    res_cropped = np.array( a, dtype=img_dst.dtype )
    
    return res_cropped

# -----------------------------------------------------------------------------
''' Image listing
    Returns a list with all images in a given folder which end in '_col.bmp'
'''
def get_imlist( path ):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('_col.bmp')]

#
class FinishFoo( Exception ):
    pass 

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    '''
    load_folder = 'D:/Projects/PillBox/Pictures/210712'
    save_folder = 'D:/Projects/PillBox/Pictures/210712_cropped'    
    '''
    load_folder = 'D:/Projects/PillBox/Pictures/tests'
    save_folder = 'D:/Projects/PillBox/Pictures/tests_cropped'

    imlist = get_imlist( load_folder )
    
    good_img = 0
    fail_img = 0

    save_to_folder = False

    try:
        start_time = time.time()
        for i in range(0,len(imlist),3):    
            img0 = cv2.imread( imlist[i] )
            img1 = cv2.imread( imlist[i+1] )
            img2 = cv2.imread( imlist[i+2] )
            # Make HDR-image
            img = np.uint8( (np.int32(img0)/3 + np.int32(img1)/3 + np.int32(img2)/3) )
            img_cropped = pic_cutout( img2, img, i, True )
            img_cropped = img_cropped*1  
    
            arr = cv.fromarray(img_cropped)
            if arr.rows>10:
                good_img += 1
                print str(i), img_cropped.shape, " --> OK"

                arr = add_scale( img_cropped ) 
                     
                if save_to_folder == True:
                    cv.SaveImage( os.path.join(save_folder, str(np.uint16(i/3))+'.jpg'), arr )
                else:                
                    cv2.imshow( 'cropped', img_cropped ) 
                    key_press = cv2.waitKey( 0 )            
                    if key_press==27:
                        print "Escaped ..."
                        raise FinishFoo
            else: 
                fail_img += 1
                print str(i), img_cropped.shape, " --> nope"     
    except FinishFoo:
        pass
                                        
    print "-------------------------------"
    print "Time = ", time.time() - start_time, " sec"
    print "Good images = ", good_img   
    print "Fail images = ", fail_img                                    
    
    cv2.waitKey(1)
    cv2.destroyAllWindows()


    # Floodfill -> does not work :)
    '''
    h,w = img_cropped.shape[:2]
    diff = (6,6,6)
    mask = np.zeros( (h+2,w+2), np.uint8 )                
    cv2.floodFill( img_cropped, mask, (10,10), (255,255,255), diff, diff )                
    '''
    