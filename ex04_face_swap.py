import cv2
import numpy
# ---------------------------------------------------------------------------------------------------------------------
from scipy.spatial import Delaunay
import detector_landmarks
import tools_calibrate
import tools_image
import tools_alg_match
# ---------------------------------------------------------------------------------------------------------------------
def apply_affine_transform(src, src_tri, target_tri, size):
    warp_mat = cv2.getAffineTransform(numpy.float32(src_tri), numpy.float32(target_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    return dst
# ---------------------------------------------------------------------------------------------------------------------
def morph_triangle(img1, img2, img, t1, t2, t, alpha):
    r1 = cv2.boundingRect(numpy.float32([t1]))
    r2 = cv2.boundingRect(numpy.float32([t2]))
    r = cv2.boundingRect(numpy.float32([t]))

    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    mask = numpy.zeros((r[3], r[2], 3), dtype=numpy.float32)
    cv2.fillConvexPoly(mask, numpy.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * (1 - mask) + img_rect * mask
# ---------------------------------------------------------------------------------------------------------------------
def get_morph(src_img,target_img,src_points,target_points,del_triangles,alpha=0.5):

    weighted_pts = []
    for i in range(0, len(src_points)):
        x = (1 - alpha) * src_points[i][0] + alpha * target_points[i][0]
        y = (1 - alpha) * src_points[i][1] + alpha * target_points[i][1]
        weighted_pts.append((x, y))

    img_morph = numpy.zeros(src_img.shape, dtype=src_img.dtype)

    for triangle in del_triangles:
        x, y, z = triangle
        t1 = [src_points[x], src_points[y], src_points[z]]
        t2 = [target_points[x], target_points[y], target_points[z]]
        t = [weighted_pts[x], weighted_pts[y], weighted_pts[z]]
        morph_triangle(src_img, target_img, img_morph, t1, t2, t, alpha)

    return img_morph
# ---------------------------------------------------------------------------------------------------------------------
def face_swap():
    folder_in = './images/ex_faceswap/'

    image1 = cv2.imread(folder_in + 'personA-2.jpg')
    image2 = cv2.imread(folder_in + 'personB-4.jpg')
    D = detector_landmarks.detector_landmarks('..//_weights//shape_predictor_68_face_landmarks.dat')
    L1 = D.get_landmarks(image1)
    L2 = D.get_landmarks(image2)

    idx = [1,2,3,4,14,15,16,17,18,19,20,21,22,23,24,25,26,27,8,9,10,32,33,34,35,36,37,38,38,40,41,42,43,44,45,46,47,48]

    H = tools_calibrate.get_transform_by_keypoints(L1[idx],L2[idx])
    aligned1, aligned2= tools_calibrate.get_stitched_images_using_translation(image1, image2, H)

    L1 = D.get_landmarks(aligned1)
    L2 = D.get_landmarks(aligned2)

    DlN = Delaunay(L1)
    del_triangles = DlN.vertices

    face1 = get_morph(aligned1,aligned2,L1,L2,del_triangles,alpha=0)
    face2 = get_morph(aligned1, aligned2, L1, L2, del_triangles, alpha=1)

    result1 = tools_image.blend_multi_band_large_small(aligned2, face1, (0, 0, 0))
    result2 = tools_image.blend_multi_band_large_small(aligned1, face2, (0, 0, 0))

    cv2.imwrite('./images/output/face1.jpg', face1)
    cv2.imwrite('./images/output/face2.jpg', face2)
    cv2.imwrite('./images/output/aligned1.jpg', aligned1)
    cv2.imwrite('./images/output/aligned2.jpg', aligned2)
    cv2.imwrite('./images/output/result1.jpg', result1)
    cv2.imwrite('./images/output/result2.jpg', result2)


    return
# ---------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    face_swap()

    #folder_in = './images/ex_faceswap/'

    #large= cv2.imread(folder_in + 'aligned.jpg')
    #mask = cv2.imread(folder_in + 'mask.jpg')

    #result = tools_image.blend_multi_band_large_small(large, mask, background_color=(0, 0, 0))
    #cv2.imwrite('./images/output/res.jpg', result)

