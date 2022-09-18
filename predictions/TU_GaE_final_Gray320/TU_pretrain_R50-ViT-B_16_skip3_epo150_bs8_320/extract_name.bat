dir *_pred.png /b/s > pred_name.txt

@echo off
for /f "tokens=*" %%a in (C:\Users\weian\Desktop\latest\predictions\TU_GaE_final_Gray320\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_320\pred_name.txt) do (
  if exist "C:\Users\weian\Desktop\latest\predictions\TU_GaE_final_Gray320\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_320\%%~nxa" (
    copy /y "C:\Users\weian\Desktop\latest\predictions\TU_GaE_final_Gray320\TU_pretrain_R50-ViT-B_16_skip3_epo150_bs8_320\%%~nxa" "C:\Users\weian\Desktop\latest\predictions\TU_GaE_final_Gray320\pure_pred"
  )
)

pause