using System;
using System.IO;
using System.Linq;
using UnityEngine;


public class Helper : MonoBehaviour
{
    
    private const string FolderPath = "MotionData";
    
    public static void SetGameObjectActive(GameObject gameObj, bool state)
    {
        if (gameObj)
        {
            gameObj.SetActive(state);
        }
    }
    
    public static GameObject CopyGameObject(GameObject original, bool worldPositionStays, string userCreatedAt)
    {
        // Create a copy using Instantiate with "false" for inactive (doesn't activate the copy)
        var copy = Instantiate(original, original.transform.parent, worldPositionStays);
        // Set the copied object's name (optional)
        copy.name = original.name + " " + userCreatedAt;
        return copy;
    }
    
    public static void SetParent(GameObject child, GameObject newParent)
    {
        if (child != null && newParent != null)
        {
            child.transform.SetParent(newParent.transform);
        }
        else
        {
            Debug.LogError("Failed to set parent! Child or new parent is null.");
        }
    }

    public static string GetID(string response)
    {
        // Find the index of "id" field
        var idIndex = response.IndexOf("\"id\"", StringComparison.Ordinal);

        if (idIndex != -1)
        {
            // Find the index of the value of "id" field
            var idValueStartIndex = response.IndexOf(':', idIndex) + 1;
            var idValueEndIndex = response.IndexOfAny(new char[] { ',', '}' }, idValueStartIndex);

            if (idValueStartIndex != -1 && idValueEndIndex != -1)
            {
                // Extract the value of "id" field as string
                var idString = response.Substring(idValueStartIndex, idValueEndIndex - idValueStartIndex);

                // Convert the string value to integer
                int id;
                if (int.TryParse(idString, out id))
                {
                    Debug.Log("ID: " + id);
                    return id.ToString();
                }
                else
                {
                    Debug.LogError("Failed to parse ID.");
                }
            }
            else
            {
                Debug.LogError("Failed to find the end of ID value.");
            }
        }
        else
        {
            Debug.LogError("Failed to find ID field.");
        }

        return "error";
    }

    public static string GetLatestFile()
    {
        var latestFile = "";
        // Check if the folder path is valid
        if (Directory.Exists(FolderPath))
        {
            // Get all files in the folder
            var files = Directory.GetFiles(FolderPath);

            // Sort the files by last write time in descending order
            Array.Sort(files, (a, b) => File.GetLastWriteTime(b).CompareTo(File.GetLastWriteTime(a)));

            // Get the latest file (first element after sorting)
            latestFile = files.FirstOrDefault();

            // Do something with the latest file
            Debug.Log("Latest file: " + latestFile);
        }
        else
        {
            Debug.LogError("Folder path does not exist: " + FolderPath);
        }
        return latestFile;
    }
    
    public static double RoundToNearestNeighbor(float number)
    {
        // Calculate the absolute value of the rounding error
        var error = Math.Abs(number - Math.Round(number, 1));

        // Round up if the error is greater than half the step size (0.05f)
        if (error > 0.05f)
        {
            return Math.Round(number, 1) + 0.1f;
        }
        else
        {
            // Otherwise, round down
            return Math.Round(number, 1);
        }
    }
    
}
