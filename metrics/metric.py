import sys
sys.path.append('rangenetpp/lidar_bonnetal_master/train/tasks/semantic')
sys.path.append('rangenetpp/lidar_bonnetal_master/train/')
import argparse
import metrics.histogram.mmd as mmd
import metrics.histogram.jsd as jsd
import os
import rangenetpp.lidar_bonnetal_master.train.tasks.semantic.infer_lib as rangenetpp
import glob
import metrics.fid.lidargen_fid as lidargen_fid
import metrics.iou as lidargen_iou
import metrics.mae as lidargen_mae
import torch

def generate_kitti_fid(folder_fid, folder_segmentations, sample_count,  seed=0):

    # Get dump for KITTI
    os.system("rm -r {folder_segmentations}".format(folder_segmentations=folder_segmentations))
    os.system("rm -r {folder_fid}".format(folder_fid=folder_fid))
    os.system("mkdir {folder_segmentations}".format(folder_segmentations=folder_segmentations))
    os.system("mkdir {folder_fid}".format(folder_fid=folder_fid))

    rangenetpp.main('--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --kitti --output_dir {folder_segmentations} --frd_dir {folder_fid} --kitti_count {kitti_count} --seed {seed}'.format(
        kitti_count=str(sample_count), seed=str(seed), folder_segmentations=folder_segmentations, folder_fid=folder_fid))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nus", help="Use nuScenes dataset", action="store_true")
    parser.add_argument("--mmd", help="Calculate MMD between samples and KITTI-360", action="store_true")
    parser.add_argument("--jsd", help="Caculate JSD between samples and KITTI-360", action="store_true")
    parser.add_argument("--exp", help="Folder of generated point cloud.")
    parser.add_argument("--fid_folder1", help="Manually provide folder1", type=str, default=None)
    parser.add_argument("--fid_folder2", help="Manually provide folder2", type=str, default=None)
    parser.add_argument("--fid", help="Run generated samples through RangeNet to get FID (KITTI only).", action="store_true")
    parser.add_argument("--iou", help="Run RangeNet++ IOU downstream comparison of LiDARGen upsampling with nearest neighbor", action="store_true")
    parser.add_argument("--accuracy", help="Run RangeNet++ accuracy", action="store_true")
    parser.add_argument("--mae", help="Get MAE (Range Representation) for upsampling with LiDARGen, Bicubic, and Nearest Neighbors", action="store_true")
    parser.add_argument("--inpainting_mae", help="Get MAE (Range Representation) for inpainting with LiDARGen", action="store_true")

    args = parser.parse_args()

    if (args.mae):
        sample_count = len(glob.glob(args.exp + '/densification_target/*.bin'))
        for idx in range(sample_count):
            result_path = (args.exp + '/densification_result/{0}.bin'.format(str(idx)))
            target_path = (args.exp + '/densification_target/{0}.bin'.format(str(idx)))
            if not os.path.exists(result_path.replace('bin', 'pth')):
                result_range = lidargen_mae.bin2range(result_path)
                torch.save(result_range, result_path.replace('bin', 'pth'))
            if not os.path.exists(target_path.replace('bin', 'pth')):
                target_range = lidargen_mae.bin2range(target_path)
                torch.save(target_range, target_path.replace('bin', 'pth'))
        lidargen_mae.calculate_mae('{exp}/'.format(exp=str(args.exp)))
    
    if args.inpainting_mae:
        sample_count = len(glob.glob(args.exp + '/inpainting_target/*.bin'))
        for idx in range(sample_count):
            result_path = (args.exp + '/inpainting_result/{0}.bin'.format(str(idx)))
            target_path = (args.exp + '/inpainting_target/{0}.bin'.format(str(idx)))
            if not os.path.exists(result_path.replace('bin', 'pth')):
                result_range = lidargen_mae.bin2range(result_path)
                torch.save(result_range, result_path.replace('bin', 'pth'))
            if not os.path.exists(target_path.replace('bin', 'pth')):
                target_range = lidargen_mae.bin2range(target_path)
                torch.save(target_range, target_path.replace('bin', 'pth'))
        lidargen_mae.calculate_inpainting_mae('{exp}/'.format(exp=str(args.exp)))


    if(args.iou):
        os.system("rm -r " + str(args.exp) + '/target_rangenet_segmentations')
        os.system("rm -r " + str(args.exp) + '/target_rangenet_fid')
        os.system("mkdir " + str(args.exp) + "/target_rangenet_segmentations")
        os.system("mkdir " + str(args.exp) + "/target_rangenet_fid")

        os.system("rm -r " + str(args.exp) + '/result_rangenet_segmentations')
        os.system("rm -r " + str(args.exp) + '/result_rangenet_fid')
        os.system("mkdir " + str(args.exp) + "/result_rangenet_segmentations")
        os.system("mkdir " + str(args.exp) + "/result_rangenet_fid")

        rangenetpp.main("--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/densification_result/ --output_dir {exp}/result_rangenet_segmentations --frd_dir {exp}/result_rangenet_fid --point_cloud".format(exp=str(args.exp)))

        rangenetpp.main('--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/densification_target/ --output_dir {exp}/target_rangenet_segmentations --frd_dir {exp}/target_rangenet_fid --point_cloud'.format(exp=str(args.exp)))


        iou = lidargen_iou.calculate_iou("{exp}/result_rangenet_segmentations".format(exp=args.exp), "{exp}/target_rangenet_segmentations".format(exp=args.exp))

        print('-------------------------------------------------')
        print('IOU Score: ' + str(iou))
        print('-------------------------------------------------')

    if (args.accuracy):
        accuracy = lidargen_iou.calculate_accuracy("{exp}/result_rangenet_segmentations".format(exp=args.exp), "{exp}/target_rangenet_segmentations".format(exp=args.exp))
        print('-------------------------------------------------')
        print('Accuracy: ' + str(accuracy))
        print('-------------------------------------------------')

    if(args.fid):
        folder1 = ""
        folder2 = ""

        if((not (args.fid_folder1 is None)) and (not (args.fid_folder2 is None))):
            folder1 = args.fid_folder1
            folder2 = args.fid_folder2

        elif (not (args.fid_folder1 is None)):
            folder1 = args.fid_folder1
            folder2 = "kitti_fid"
            folder_segmentations = "kitti_seg"
            generate_kitti_fid(folder2, folder_segmentations, 1000, 0)

        else:
            # Get dump for model samples
            os.system("rm -r {exp}/unconditional_fid".format(exp=args.exp))
            os.system("mkdir {exp}/unconditional_fid".format(exp=args.exp))
            os.system("rm -r {exp}/unconditional_segmentations".format(exp=args.exp))
            os.system("mkdir {exp}/unconditional_segmentations".format(exp=args.exp))
            rangenetpp.main('--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/ --output_dir {exp}/unconditional_segmentations --frd_dir {exp}/unconditional_fid --point_cloud'.format(exp=str(args.exp)))
            folder1 = "{exp}/unconditional_fid/".format(exp=str(args.exp))
            # Get dump for model samples

            if not args.fid_folder2:
                sample_count = len(glob.glob("{exp}/unconditional_segmentations/*".format(exp=args.exp)))
            
                # Get dump for KITTI
                folder_kitti_seg = "kitti_segmentations"
                folder_kitti_fid = "kitti_fid"
                generate_kitti_fid(folder_kitti_fid, folder_kitti_seg, sample_count, 0)

                folder2 = folder_kitti_fid 
            else:
                folder2 = args.fid_folder2

        fid_score = lidargen_fid.get_fid(folder1, folder2)

        print('-------------------------------------------------')
        print('FID Score: ' + str(fid_score))
        print('-------------------------------------------------')
        
    if (args.jsd):
        jsd_score = jsd.calculate_jsd(args.exp, from_bin=True) \
                    if not args.nus else jsd.calculate_jsd_nus(args.exp, from_bin=True)

        print('-------------------------------------------------')
        print('JSD Score: ' + str(jsd_score))
        print('-------------------------------------------------')

    if (args.mmd):
        mmd_score = mmd.calculate_mmd(args.exp, from_bin=True) \
                    if not args.nus else mmd.calculate_mmd_nus(args.exp, from_bin=True)

        print('-------------------------------------------------')
        print('MMD Score: ' + str(mmd_score))
        print('-------------------------------------------------')

if __name__ == '__main__':
    main()