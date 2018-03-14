
Adapted from https://github.com/aiunderstand/unity-cntk/blob/master/README.md.


1) Download the released version from - CNTK GPU 2.3 https://cntk.ai/dlwg-2.3.html (or see https://github.com/Microsoft/CNTK/releases for latest version)

2) Copy the Cntk.Core.Managed-VERSION.dll to you Assets folder. Recommanded somewhere under Plugins folder.

2) Copy the following DLL's to where the system can find them but not in the Assets folder. Rocommanded in the project folder.

[some others might be needed as well. Just add new dlls based on the error message from Unity if you encounter any.]



(see https://github.com/Microsoft/CNTK/wiki/CNTK-Library-Evaluation-on-Windows#using-the-cntk-library-managed-api 
and http://stackoverflow.com/questions/36527985/dllnotfoundexception-in-while-building-desktop-unity-application-using-artoolkit)

  - Cntk.Core-VERSION.dll (eg. Cntk.Core-2.3.dll)
  - Cntk.Math-VERSION.dll
  - Cntk.Core.CSBinding-VERSION.dll
  - Cntk.Composite-VERSION.dll
  - Cntk.PerformanceProfiler-VERSION.dll
  - Cntk.Deserializers.TextFormat-VERSION.dll  
  - libiomp5md.dll
  - mklml.dll
  - cublas64_80.dll
  - cudart64_80.dll
  - cudnn64_5.dll
  - curand64_80.dll
  - cusparse64_80.dll
  - nvml.dll