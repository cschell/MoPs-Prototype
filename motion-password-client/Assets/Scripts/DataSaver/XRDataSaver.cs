using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.XR;


[RequireComponent(typeof(XRTrackableObjects))]
public class XRDataSaver : MonoBehaviour
{
    public string pathToCsv = "MotionData"; // Path to the CSV File
    
    public int chunkSizeInMilliseconds = 500;
    
    private static XRDataSaver _xrDataSaver;
    private XRTrackableObjects _xrTrackableObjects;
    
    private string _dateTime;
    private bool _isRecording;

    private readonly Queue<string> _dataRows = new();
    
    private Coroutine _saveDataRoutine;
    private Coroutine _addDataRoutine;
    
    private void Awake()
    {
        
        if (_xrDataSaver != null)
        {
            Destroy(gameObject);
            return;
        }
        
        _xrDataSaver = this;
        _isRecording = false;
        DontDestroyOnLoad(this);
        
        _xrTrackableObjects = GetComponent<XRTrackableObjects>();
    }

    private void OnDestroy()
    {
        if (_xrDataSaver != this) return;
        if (_addDataRoutine != null)
        {
            StopCoroutine(_addDataRoutine);
        }
        if (_saveDataRoutine != null)
        {
            StopCoroutine(_saveDataRoutine);
        }
        SaveAllData();
    }
    
    public void StartRecording()
    {
        if (_isRecording) return;
        UpdateTimeCsvPath();
        _addDataRoutine = StartCoroutine(AddDataCoroutine());
        _saveDataRoutine = StartCoroutine(SaveDataCoroutine());
        _isRecording = true;
    }

    public void StopRecording()
    {
        if (!_isRecording) return;
        StopCoroutine(_addDataRoutine);
        StopCoroutine(_saveDataRoutine);
        SaveAllData();
        _isRecording = false;
    }

    public bool IsRecording()
    {
        return _isRecording;
    }

    public void UpdateTimeCsvPath()
    {
        _dateTime = DateTime.Now.ToString("yyyy-M-d_HH-mm-ss");
        pathToCsv = Path.Join("MotionData", $"movement_{_dateTime}.csv");
        
        if (!File.Exists(pathToCsv))
        {
            AddHeaderLine();
        }
    }

    public void UpdateChunkSize(int newChunkSizeInSeconds)
    {
        chunkSizeInMilliseconds = newChunkSizeInSeconds;
    }
    
    private void SaveAllData()
    {
        if (_dataRows.Count <= 0) return;

        var currentRowCount = _dataRows.Count;
        using var writer = new StreamWriter(pathToCsv, true);
        for (var i = 0; i < currentRowCount; i++)
        {
            writer.WriteLine(_dataRows.Dequeue());
        }
    }


    private void AddHeaderLine()
    {
        using var writer = new StreamWriter(pathToCsv, true);
        //// Write the header line
        writer.WriteLine(
            "Timestamp,Frame,RealtimeSinceStartup,UserPresence,IsTracked,TrackingState," +
            "hmd_pos_x,hmd_pos_y,hmd_pos_z," +
            "hmd_rot_x,hmd_rot_y,hmd_rot_z,hmd_rot_w," +
            "hmd_velocity_x,hmd_velocity_y,hmd_velocity_z," +
            "hmd_angular_velocity_x,hmd_angular_velocity_y,hmd_angular_velocity_z," +
            "Center_eye_pos_x,Center_eye_pos_y,Center_eye_pos_z," +
            "Center_eye_rot_x,Center_eye_rot_y,Center_eye_rot_z,Center_eye_rot_w," +
            "Center_eye_velocity_x,Center_eye_velocity_y,Center_eye_velocity_z," +
            "Center_eye_angular_velocity_x,Center_eye_angular_velocity_y,Center_eye_angular_velocity_z," +
            "left_eye_pos_x,left_eye_pos_y,left_eye_pos_z," +
            "left_eye_rot_x,left_eye_rot_y,left_eye_rot_z,left_eye_rot_w," +
            "left_eye_velocity_x,left_eye_velocity_y,left_eye_velocity_z," +
            "left_eye_angular_velocity_x,left_eye_angular_velocity_y,left_eye_angular_velocity_z," +
            "right_eye_pos_x,right_eye_pos_y,right_eye_pos_z," +
            "right_eye_rot_x,right_eye_rot_y,right_eye_rot_z,right_eye_rot_w," +
            "right_eye_velocity_x,right_eye_velocity_y,right_eye_velocity_z," +
            "right_eye_angular_velocity_x,right_eye_angular_velocity_y,right_eye_angular_velocity_z," +
            "right_controller_Primary2DA_xis_x,right_controller_Primary2DA_xis_y,right_controller_Grip," +
            "right_controller_Grip_button_,right_controller_Menu_button_,right_controller_Primary_button_," +
            "right_controller_PrimaryTouch,right_controller_Secondary_button_,right_controller_SecondaryTouch," +
            "right_controller_Trigger,right_controller_Trigger_button_,right_controller_Primary2DA_xisClick," +
            "right_controller_Primary2DA_xisTouch,right_controller_IsTracked,right_controller_TrackingState," +
            "right_controller_pos_x,right_controller_pos_y,right_controller_pos_z," +
            "right_controller_rot_x,right_controller_rot_y,right_controller_rot_z,right_controller_rot_w," +
            "right_controller_velocity_x,right_controller_velocity_y,right_controller_velocity_z," +
            "right_controller_angular_velocity_x,right_controller_angular_velocity_y,right_controller_angular_velocity_z," +
            "right_controller_pointer_pos_x,right_controller_pointer_pos_y,right_controller_pointer_pos_z," +
            "right_controller_pointer_rot_x,right_controller_pointer_rot_y,right_controller_pointer_rot_z,right_controller_pointer_rot_w," +
            "right_controller_pointer_velocity_x,right_controller_pointer_velocity_y,right_controller_pointer_velocity_z," +
            "right_controller_pointer_angular_velocity_x,right_controller_pointer_angular_velocity_y,right_controller_pointer_angular_velocity_z," +
            "left_controller_Primary2DA_xis_x,left_controller_Primary2DA_xis_y,left_controller_Grip," +
            "left_controller_Grip_button_,left_controller_Menu_button_,left_controller_Primary_button_," +
            "left_controller_PrimaryTouch,left_controller_Secondary_button_,left_controller_SecondaryTouch," +
            "left_controller_Trigger,left_controller_Trigger_button_,left_controller_Primary2DA_xisClick," +
            "left_controller_Primary2DA_xisTouch,left_controller_IsTracked,left_controller_TrackingState," +
            "left_controller_pos_x,left_controller_pos_y,left_controller_pos_z," +
            "left_controller_rot_x,left_controller_rot_y,left_controller_rot_z,left_controller_rot_w," +
            "left_controller_velocity_x,left_controller_velocity_y,left_controller_velocity_z," +
            "left_controller_angular_velocity_x,left_controller_angular_velocity_y,left_controller_angular_velocity_z," +
            "left_controller_pointer_pos_x,left_controller_pointer_pos_y,left_controller_pointer_pos_z," +
            "left_controller_pointer_rot_x,left_controller_pointer_rot_y,left_controller_pointer_rot_z,left_controller_pointer_rot_w," +
            "left_controller_pointer_velocity_x,left_controller_pointer_velocity_y,left_controller_pointer_velocity_z," +
            "left_controller_pointer_angular_velocity_x,left_controller_pointer_angular_velocity_y,left_controller_pointer_angular_velocity_z,");
    }

    private IEnumerator SaveDataCoroutine()
    {
        while (true)
        {
            yield return new WaitForSecondsRealtime(chunkSizeInMilliseconds / 1000f);

            SaveAllData();
        }
        // ReSharper disable once IteratorNeverReturns
    }

    private IEnumerator AddDataCoroutine()
    {
        while (true)
        {
            var (xrHeadDataRow, xrRightControllerDataRow, xrLeftControllerDataRow) = GatherDataIntoLists();

            if (xrHeadDataRow.IsTracked.Equals("True") && xrRightControllerDataRow.IsTracked.Equals("True") &&
                xrLeftControllerDataRow.IsTracked.Equals("True"))
                AddDataToRows(xrHeadDataRow, xrRightControllerDataRow, xrLeftControllerDataRow);

            yield return null;
        }
        // ReSharper disable once IteratorNeverReturns
    }

    private (XRHeadDataRow, XRControllerDataRow, XRControllerDataRow)  GatherDataIntoLists()
    {       

        XRHeadDataRow xrHeadDataRow = new XRHeadDataRow();

        xrHeadDataRow = GetHeadData(xrHeadDataRow, _xrTrackableObjects.Hmd);

    
        XRControllerDataRow xrRightControllerDataRow = new XRControllerDataRow();

        xrRightControllerDataRow = GetControllerData(xrRightControllerDataRow, _xrTrackableObjects.RightController);

    
        XRControllerDataRow xrLeftControllerDataRow = new XRControllerDataRow();

        xrLeftControllerDataRow = GetControllerData(xrLeftControllerDataRow, _xrTrackableObjects.LeftController);
    
        return (xrHeadDataRow, xrRightControllerDataRow, xrLeftControllerDataRow);
    }

    private XRHeadDataRow GetHeadData(XRHeadDataRow xrHeadDataRow, InputDevice inputDevice)
    {
        xrHeadDataRow.UserPresence = GetFeatureValue(inputDevice, CommonUsages.userPresence);
        xrHeadDataRow.IsTracked = GetFeatureValue(inputDevice, CommonUsages.isTracked);
        xrHeadDataRow.TrackingState = GetFeatureValue(inputDevice, CommonUsages.trackingState);
        xrHeadDataRow.DevicePosition = GetFeatureValue(inputDevice, CommonUsages.devicePosition);
        xrHeadDataRow.DeviceRotation = GetFeatureValue(inputDevice, CommonUsages.deviceRotation);
        xrHeadDataRow.DeviceVelocity = GetFeatureValue(inputDevice, CommonUsages.deviceVelocity);
        xrHeadDataRow.DeviceAngularVelocity = GetFeatureValue(inputDevice, CommonUsages.deviceAngularVelocity);
        xrHeadDataRow.CenterEyePosition = GetFeatureValue(inputDevice, CommonUsages.centerEyePosition);
        xrHeadDataRow.CenterEyeRotation = GetFeatureValue(inputDevice, CommonUsages.centerEyeRotation);
        xrHeadDataRow.CenterEyeVelocity = GetFeatureValue(inputDevice, CommonUsages.centerEyeVelocity);
        xrHeadDataRow.CenterEyeAngularVelocity = GetFeatureValue(inputDevice, CommonUsages.centerEyeAngularVelocity);
        xrHeadDataRow.LeftEyePosition = GetFeatureValue(inputDevice, CommonUsages.leftEyePosition);
        xrHeadDataRow.LeftEyeRotation = GetFeatureValue(inputDevice, CommonUsages.leftEyeRotation);
        xrHeadDataRow.LeftEyeVelocity = GetFeatureValue(inputDevice, CommonUsages.leftEyeVelocity);
        xrHeadDataRow.LeftEyeAngularVelocity = GetFeatureValue(inputDevice, CommonUsages.leftEyeAngularVelocity);
        xrHeadDataRow.RightEyePosition = GetFeatureValue(inputDevice, CommonUsages.rightEyePosition);
        xrHeadDataRow.RightEyeRotation = GetFeatureValue(inputDevice, CommonUsages.rightEyeRotation);
        xrHeadDataRow.RightEyeVelocity = GetFeatureValue(inputDevice, CommonUsages.rightEyeVelocity);
        xrHeadDataRow.RightEyeAngularVelocity = GetFeatureValue(inputDevice, CommonUsages.rightEyeAngularVelocity);
    
        return xrHeadDataRow;
    }

    private XRControllerDataRow GetControllerData(XRControllerDataRow xrControllerDataRow, InputDevice inputDevice)
    {
        xrControllerDataRow.Primary2DAxis = GetFeatureValue(inputDevice, CommonUsages.primary2DAxis);
        xrControllerDataRow.Grip = GetFeatureValue(inputDevice, CommonUsages.grip);
        xrControllerDataRow.GripButton = GetFeatureValue(inputDevice, CommonUsages.gripButton);
        xrControllerDataRow.MenuButton = GetFeatureValue(inputDevice, CommonUsages.menuButton);

        xrControllerDataRow.PrimaryButton = GetFeatureValue(inputDevice, CommonUsages.primaryButton);
        xrControllerDataRow.PrimaryTouch = GetFeatureValue(inputDevice, CommonUsages.primaryTouch);
        xrControllerDataRow.SecondaryButton = GetFeatureValue(inputDevice, CommonUsages.secondaryButton);
        xrControllerDataRow.SecondaryTouch = GetFeatureValue(inputDevice, CommonUsages.secondaryTouch);

        xrControllerDataRow.Trigger = GetFeatureValue(inputDevice, CommonUsages.trigger);
        xrControllerDataRow.TriggerButton = GetFeatureValue(inputDevice, CommonUsages.triggerButton);
    
        xrControllerDataRow.Primary2DAxisClick = GetFeatureValue(inputDevice, CommonUsages.primary2DAxisClick);
        xrControllerDataRow.Primary2DAxisTouch = GetFeatureValue(inputDevice, CommonUsages.primary2DAxisTouch);

        xrControllerDataRow.IsTracked = GetFeatureValue(inputDevice, CommonUsages.isTracked);
        xrControllerDataRow.TrackingState = GetFeatureValue(inputDevice, CommonUsages.trackingState);

        xrControllerDataRow.DevicePosition = GetFeatureValue(inputDevice, CommonUsages.devicePosition);
        xrControllerDataRow.DeviceRotation = GetFeatureValue(inputDevice, CommonUsages.deviceRotation);
        xrControllerDataRow.DeviceVelocity = GetFeatureValue(inputDevice, CommonUsages.deviceVelocity);
        xrControllerDataRow.DeviceAngularVelocity = GetFeatureValue(inputDevice, CommonUsages.deviceAngularVelocity);
    
        xrControllerDataRow.PointerPosition = GetFeatureValue(inputDevice, CommonUsages.devicePosition);
        xrControllerDataRow.PointerRotation = GetFeatureValue(inputDevice, CommonUsages.deviceRotation);
        xrControllerDataRow.PointerVelocity = GetFeatureValue(inputDevice, CommonUsages.deviceVelocity);
        xrControllerDataRow.PointerAngularVelocity = GetFeatureValue(inputDevice, CommonUsages.deviceAngularVelocity);

        return xrControllerDataRow;
    }

    private void AddDataToRows(XRHeadDataRow xrHeadDataRow, XRControllerDataRow xrRightControllerDataRow, XRControllerDataRow xrLeftControllerDataRow)
    {
        StringBuilder stringBuilder = new StringBuilder();
    
        // Append data fields from xrHeadDataRow
        var cultureInfo = CultureInfo.InvariantCulture;
        stringBuilder.Append(DateTime.Now.ToString("T", cultureInfo));
        stringBuilder.Append(",");
        stringBuilder.Append(Time.frameCount.ToString(cultureInfo));
        stringBuilder.Append(",");
        stringBuilder.Append(Time.realtimeSinceStartup.ToString(cultureInfo));
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.UserPresence);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.IsTracked);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.TrackingState);
        stringBuilder.Append(",");

        stringBuilder.Append(xrHeadDataRow.DevicePosition);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.DeviceRotation);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.DeviceVelocity);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.DeviceAngularVelocity);
        stringBuilder.Append(",");

        stringBuilder.Append(xrHeadDataRow.CenterEyePosition);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.CenterEyeRotation);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.CenterEyeVelocity);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.CenterEyeAngularVelocity);
        stringBuilder.Append(",");

        stringBuilder.Append(xrHeadDataRow.LeftEyePosition);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.LeftEyeRotation);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.LeftEyeVelocity);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.LeftEyeAngularVelocity);
        stringBuilder.Append(",");

        stringBuilder.Append(xrHeadDataRow.RightEyePosition);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.RightEyeRotation);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.RightEyeVelocity);
        stringBuilder.Append(",");
        stringBuilder.Append(xrHeadDataRow.RightEyeAngularVelocity);
        stringBuilder.Append(",");


        // Append data fields from xrRightControllerDataRow
        stringBuilder.Append(xrRightControllerDataRow.Primary2DAxis);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.Grip);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.GripButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.MenuButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.PrimaryButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.PrimaryTouch);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.SecondaryButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.SecondaryTouch);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.Trigger);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.TriggerButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.Primary2DAxisClick);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.Primary2DAxisTouch);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.IsTracked);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.TrackingState);
        stringBuilder.Append(",");

        stringBuilder.Append(xrRightControllerDataRow.DevicePosition);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.DeviceRotation);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.DeviceVelocity);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.DeviceAngularVelocity);
        stringBuilder.Append(",");

        stringBuilder.Append(xrRightControllerDataRow.PointerPosition);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.PointerRotation);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.PointerVelocity);
        stringBuilder.Append(",");
        stringBuilder.Append(xrRightControllerDataRow.PointerAngularVelocity);
        stringBuilder.Append(",");

        // Append data fields from xrLeftControllerDataRow
        stringBuilder.Append(xrLeftControllerDataRow.Primary2DAxis);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.Grip);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.GripButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.MenuButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.PrimaryButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.PrimaryTouch);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.SecondaryButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.SecondaryTouch);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.Trigger);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.TriggerButton);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.Primary2DAxisClick);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.Primary2DAxisTouch);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.IsTracked);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.TrackingState);
        stringBuilder.Append(",");
    
        stringBuilder.Append(xrLeftControllerDataRow.DevicePosition);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.DeviceRotation);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.DeviceVelocity);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.DeviceAngularVelocity);
        stringBuilder.Append(",");

        stringBuilder.Append(xrLeftControllerDataRow.PointerPosition);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.PointerRotation);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.PointerVelocity);
        stringBuilder.Append(",");
        stringBuilder.Append(xrLeftControllerDataRow.PointerAngularVelocity);
        stringBuilder.Append(",");

        _dataRows.Enqueue(stringBuilder.ToString());
    }

    private static string GetFeatureValue(InputDevice inputDevice, InputFeatureUsage<float> grip)
    {
        return inputDevice.TryGetFeatureValue(grip, out var value) ? value.ToString(CultureInfo.InvariantCulture).Replace(",", ".") : string.Empty;
    }

    private static string GetFeatureValue(InputDevice inputDevice, InputFeatureUsage<InputTrackingState> trackingState)
    {
        return inputDevice.TryGetFeatureValue(trackingState, out var value) ? value.ToString().Replace(",", ";") : string.Empty;
    }

    private static string GetFeatureValue(InputDevice inputDevice, InputFeatureUsage<bool> usage)
    {
        return inputDevice.TryGetFeatureValue(usage, out var value) ? value.ToString() : string.Empty;
    }

    private static string GetFeatureValue(InputDevice inputDevice, InputFeatureUsage<Vector2> primary2DAxis)
    {
        return inputDevice.TryGetFeatureValue(primary2DAxis, out var vector) ? $"{vector.x}".Replace(",", ".") + 
        "," + 
        $"{vector.y}".Replace(",", ".") 
        : string.Empty;
    }

    private static string GetFeatureValue(InputDevice inputDevice, InputFeatureUsage<Vector3> usage)
    {
        return inputDevice.TryGetFeatureValue(usage, out var vector) ? $"{vector.x}".Replace(",", ".") + 
        "," + 
        $"{vector.y}".Replace(",", ".") + 
        "," + 
        $"{vector.z}".Replace(",", ".") 
        : string.Empty;
    }

    private static string GetFeatureValue(InputDevice inputDevice, InputFeatureUsage<Quaternion> usage)
    {
        return inputDevice.TryGetFeatureValue(usage, out var quaternion) ? $"{quaternion.x}".Replace(",", ".") + 
        "," +
        $"{quaternion.y}".Replace(",", ".") +
        "," +
        $"{quaternion.z}".Replace(",", ".") +
        "," +
        $"{quaternion.w}".Replace(",", ".")
        : string.Empty;
    }
   
    private class DataRow{}    

    private class XRHeadDataRow : DataRow
    {
        public string UserPresence { get; set; }
        public string IsTracked { get; set; }
        public string TrackingState { get; set; }
        public string DevicePosition { get; set; }
        public string DeviceRotation { get; set; }
        public string DeviceVelocity { get; set; }
        public string DeviceAngularVelocity { get; set; }
        public string CenterEyePosition { get; set; }
        public string CenterEyeRotation { get; set; }
        public string CenterEyeVelocity { get; set; }
        public string CenterEyeAngularVelocity { get; set; }
        public string LeftEyePosition { get; set; }
        public string LeftEyeRotation { get; set; }
        public string LeftEyeVelocity { get; set; }
        public string LeftEyeAngularVelocity { get; set; }
        public string RightEyePosition { get; set; }
        public string RightEyeRotation { get; set; }
        public string RightEyeVelocity { get; set; }
        public string RightEyeAngularVelocity { get; set; }

    }

    private class XRControllerDataRow : DataRow
    {
        public string Primary2DAxis { get; set; }
        public string Grip { get; set; }
        public string GripButton { get; set; }
        public string MenuButton { get; set; }
        public string PrimaryButton { get; set; }
        public string PrimaryTouch { get; set; }
        public string SecondaryButton { get; set; }
        public string SecondaryTouch { get; set; }
        public string Trigger { get; set; }
        public string TriggerButton { get; set; }
        public string TriggerTouch { get; set; }
        public string Primary2DAxisClick { get; set; }
        public string Primary2DAxisTouch { get; set; }
        public string IsTracked { get; set; }
        public string TrackingState { get; set; }
        public string DevicePosition { get; set; }
        public string DeviceRotation { get; set; }
        public string DeviceVelocity { get; set; }
        public string DeviceAngularVelocity { get; set; }
        public string PointerPosition { get; set; }
        public string PointerRotation { get; set; }
        public string PointerVelocity { get; set; }
        public string PointerAngularVelocity { get; set; }

    }

}


