
import scipy.io
mat = scipy.io.loadmat('synthesizedMotion.mat')

import matplotlib.pyplot as plt 
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 18})

nFiles = mat.get("nFiles")[0][0]
allMotionTrajectory = mat.get("allMotionTrajectory")
for iMotion in range (nFiles): #(nFiles):
    print(iMotion+1)
    fig = plt.figure()
    #plt.plot(allMotionTrajectory[iMotion,:,0], linewidth = 2,label = 'X-Trans')
    #plt.plot(allMotionTrajectory[iMotion,:,1], linewidth = 2,label = 'Y-Trans')
    plt.plot(allMotionTrajectory[iMotion,:,5], linewidth = 2,label = 'Z-Rotation')
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend()
    #plt.gca().set_aspect('equal', adjustable='box')
    # square plot
    #plt.axis('square')

    motion_str_fn = "Exp_21_S8_Motion_File_Num_{}.png".format(iMotion+1)
    plt.savefig(motion_str_fn, dpi=600)
    
#     close all;
#     figure;
#     set(gcf,'Color','w');
#     hold on;
#     title(['Motion #', num2str(iMotion)])
#     set(gca,'FontSize',25);
#     plot(squeeze(allMotionTrajectory(iMotion,:,1)),'LineWidth',1.5);
#     plot(squeeze(allMotionTrajectory(iMotion,:,2)),'LineWidth',1.5);
#     plot(squeeze(allMotionTrajectory(iMotion,:,6)),'LineWidth',1.5);
#     legend('X-Trans','Y-Trans','Z-Rotation');
#     pause;
# end
