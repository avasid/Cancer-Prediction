path =  "/media/aakif/Common/MATLAB_files_both/";
save_path = "/media/aakif/Common/MATLAB_extract_both/";
Patients = dir(path);
nPatients = length(Patients);
for i = 3:nPatients
    
    load(path + Patients(i).name);
%     j = 0;
    disp(Patients(i).name);
    ROIbox = permute(ROIbox, [2 3 1]);
    mask = permute(mask, [2 3 1]);
%     mask(isnan(mask))=0;
    ROIbox = double(ROIbox);
    ROIonly = ROIbox; 
    ROIonly(isnan(mask)) = NaN;
%     ROIonly(mask<0) = NaN;
%%%% Morphological features
    [eccentricity] = getEccentricity(ROIonly,pixelW,sliceS);
    [sizeROI] = getSize(ROIonly,pixelW,sliceS);
    [solidity] = getSolidity(ROIonly,pixelW,sliceS);
    [volume] = getVolume(ROIonly,pixelW,sliceS);
    
    for Ng = [8 16 32 64]
        levels = 1:Ng;
        for voxel =1:5
            scale0 = pixelW/voxel;
            scale1 = sliceS/voxel;
            tsize = [round(double(size(ROIbox,1))*scale0),round(double(size(ROIbox,2))*scale0),...
                round(double(size(ROIbox,3))*scale1)];
            re_mask = imresize3D(mask,[],tsize,'nearest','fill');
            re_ROIbox = imresize3D(ROIbox,[],tsize,'cubic','fill');
%             re_arr = imresize3D(arr,[],tsize,'cubic','fill');
            re_ROIonly = re_ROIbox; 
            re_ROIonly(isnan(re_mask)) = NaN;
%             re_ROIonly(re_mask<0) = NaN;
            if length(~isnan(re_ROIonly(:)))<2
                disp("Problem1"+voxel+Ng);
                disp("eln=" + length(~isnan(re_ROIonly(:))));
            end
            for norm = 1:2
                if norm == 1
                    [ROIonly_quan,levels] = uniformQuantization(re_ROIonly,Ng);
                else
                    [ROIonly_quan,levels] = equalQuantization(re_ROIonly,Ng);
                end
                if length(~isnan(ROIonly_quan(:)))<2
                    disp("Problem2"+voxel+Ng+norm);
                    disp("eln=" + length(~isnan(ROIonly_quan(:))));
                end
                [textures_NGTDM,textures_GLSZM,textures_GLRLM,textures_GLCM,textures_Global] = Main(ROIonly_quan,levels);
                save(save_path + Patients(i).name + "_"+norm+"_"+voxel+"_"+Ng,...
                'textures_NGTDM','textures_GLSZM','textures_GLRLM','textures_GLCM',...
                'textures_Global','eccentricity','sizeROI','solidity','volume');
%                 j = j+1;
%                 disp(j);
            end
        end
    end
    
        
                
    
    
    clearvars -except i path Patients nPatients save_path



end

