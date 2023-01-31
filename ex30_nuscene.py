import numpy
import pandas as pd
from nuscenes import NuScenes
# ----------------------------------------------------------------------------------------------------------------------
import tools_wavefront
from CV import tools_pr_geom
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
folder_out = './images/output/'
# ----------------------------------------------------------------------------------------------------------------------
def export_pointcloud_to_df():
    nusc = NuScenes(version='v1.0-mini', dataroot='./images/ex_nuscenes2/', verbose=True)
    ll = [i for i, x in enumerate(nusc.sample) if x['timestamp'] == 1533151604048025][0]
    my_sample = nusc.sample[ll]

    sample_record = nusc.get('sample', my_sample['token'])
    pointsensor_token = sample_record['data']['LIDAR_TOP']
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = nusc.dataroot+ pointsensor['filename']
    points = numpy.fromfile(pcl_path, dtype=numpy.float32).reshape((-1, 5))

    df = pd.DataFrame(points[:,:3])
    return df
# ----------------------------------------------------------------------------------------------------------------------
def render_pointcloud():
    nusc = NuScenes(version='v1.0-mini', dataroot='./images/ex_nuscenes2/', verbose=True)
    ll = [i for i,x in enumerate(nusc.sample) if x['timestamp']==1533151604048025][0]
    my_sample = nusc.sample[ll]
    sample_data_token = my_sample['data']['LIDAR_TOP']
    nusc.get_sample_lidarseg_stats(my_sample['token'], sort_by='count')


    nusc.render_pointcloud_in_image(my_sample['token'],pointsensor_channel='LIDAR_TOP',camera_channel='CAM_FRONT_LEFT',render_intensity=False,
                                    show_lidarseg=True,filter_lidarseg_labels=[22, 23, 24],show_lidarseg_legend=True,
                                    out_path='./images/output/')

    return
# ----------------------------------------------------------------------------------------------------------------------
def render_scene():
    # my_scene = nusc.scene[1]
    # nusc.render_scene_channel_lidarseg(my_scene['token'],'CAM_BACK',filter_lidarseg_labels=[18, 28],verbose=True,dpi=100,imsize=(1280, 720))
    return
# ----------------------------------------------------------------------------------------------------------------------
def random_orto_tripple(r=0.15):
    rvec = numpy.array([numpy.pi / 2, 0, 0])
    v1 = numpy.ones(3) * r
    v2 = tools_pr_geom.apply_rotation(rvec, v1)[0]
    v3 = numpy.cross(v1, v2)
    res = numpy.concatenate([v1, v2, v3]).reshape(((-1,3)))
    return res
# ----------------------------------------------------------------------------------------------------------------------
def pointcloud_df_to_obj(df):

    # df['a'] = numpy.array([180*numpy.arctan(v[0]/v[1])/numpy.pi for v in df.values])
    # df = tools_DF.apply_filter(df,'a',[0,4])
    # idx = numpy.random.choice(df.shape[0], 10000, replace=True)
    # df = df.iloc[idx]

    df_noise = pd.DataFrame(numpy.array([random_orto_tripple()]*df.shape[0]).reshape((-1, 3)))
    df3 = pd.concat([df,df,df],axis=0)
    df3['i'] = numpy.concatenate([numpy.arange(0, 3*df.shape[0], 3), numpy.arange(1, 3*df.shape[0], 3), numpy.arange(2, 3*df.shape[0], 3)])
    df3 = df3.sort_values(by='i')
    df3.iloc[:,:3]+=df_noise.iloc[:,:3].values

    idx_vertex3 = numpy.arange(0,df3.shape[0]).reshape((-1,3))
    coord_texture3 = numpy.array([numpy.linalg.norm(v[:3]) for v in df3.values])
    coord_texture3-= numpy.min(coord_texture3)
    coord_texture3 = coord_texture3/numpy.max(coord_texture3)
    coord_texture3 = numpy.concatenate([coord_texture3.reshape((-1,1)),numpy.full((coord_texture3.shape[0],1),0.95)],axis=1)

    object = tools_wavefront.ObjLoader()
    object.export_mesh('./images/ex_GL/nuscene/lidar3.obj', df3.values, coord_texture=coord_texture3, idx_vertex=idx_vertex3, do_transform=False, filename_material='lidar.mtl')

    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # df = export_pointcloud_to_df()
    # df.to_csv(folder_out+'./df_lidar.csv',index=False)
    # df = pd.read_csv(folder_out+'./df_lidar.csv')
    # pointcloud_df_to_obj(df)

    render_pointcloud()