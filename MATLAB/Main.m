function [textures_NGTDM,textures_GLSZM,textures_GLRLM,textures_GLCM,textures_Global] = Main(arr,levels)
    



    % dicompath = "/home/aakif/HeadNeck/CT/HN-CHUM-035"
    % sData = readDICOMdir(dicompath,true);
    % roi = getROIonly(sData,26);
    ROIonlyUniform = arr;

    % ROIonlyNorm = CollewetNorm(roi);
    % ROIboxFill = fillBox(roi);
    [NGTDM,countValid] = getNGTDM(ROIonlyUniform,levels);
    textures_NGTDM= getNGTDMtextures(NGTDM,countValid);
    GLSZM = getGLSZM(ROIonlyUniform,levels);
    textures_GLSZM = getGLSZMtextures(GLSZM);
    GLRLM = getGLRLM(ROIonlyUniform,levels);
    textures_GLRLM= getGLRLMtextures(GLRLM);
    GLCM= getGLCM(ROIonlyUniform,levels);
    textures_GLCM= getGLCMtextures(GLCM);
    textures_Global = getGlobalTextures(ROIonlyUniform,100);
    % sliceS= sData{2}.scan.sliceS;
    % pixelW=sData{2}.scan.pixelW;

end



