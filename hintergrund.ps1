$replacementStrings = @(
	"Prompt 1",
	"Prompt 2".
	"Prompt 3, 4, ..." 
	)

$negprompt = "Negative prompt like broken hands and such"

$randomIndex = Get-Random -Minimum 0 -Maximum $replacementStrings.Length
$prompt = $replacementStrings[$randomIndex]
$combined = $prompt +"|"+$negprompt
$code = @' 
using System.Runtime.InteropServices; 
namespace Win32{ 
    
     public class Wallpaper{ 
        [DllImport("user32.dll", CharSet=CharSet.Auto)] 
         static extern int SystemParametersInfo (int uAction , int uParam , string lpvParam , int fuWinIni) ; 
         
         public static void SetWallpaper(string thePath){ 
            SystemParametersInfo(20,0,thePath,3); 
         }
    }
 } 
'@
add-type $code
$apiUrl = "<API URL HERE>/infer/$combined"
$outputDirectory = "<PATH TO STORE THE IMAGES HERE>"
$counter = (Get-ChildItem -Path $outputDirectory | Measure-Object).Count+1
$response = Invoke-RestMethod -Uri $apiUrl -Method Get -OutFile "$outputDirectory\output$counter.png"
$value = "$outputDirectory\output$counter.png"
[Win32.Wallpaper]::SetWallpaper($value)
