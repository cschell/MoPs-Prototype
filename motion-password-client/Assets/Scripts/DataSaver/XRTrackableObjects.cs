using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;


public class XRTrackableObjects : MonoBehaviour
{
    public InputDevice RightController;
    public InputDevice LeftController;
    public InputDevice Hmd;
    private static XRTrackableObjects instance;

    void Update()
    {
        if (!RightController.isValid || !LeftController.isValid || !Hmd.isValid)
            InitializeInputDevices();
    }

    private void Awake()
    {
        if (instance != null)
        {
            Destroy(gameObject);
        }
        else
        {
            instance = this;
            DontDestroyOnLoad(this);
        }
    }

    private void Start()
    {
        if (RightController.isValid)
        {
            Debug.Log("Right Valid Device");
        } else
        {
            Debug.Log("Right NOT Valid Device");
        }
    
        if (LeftController.isValid)
        {
            Debug.Log("Left Valid Device");
        } else
        {
            Debug.Log("Left NOT Valid Device");
        }
    }

    private void ValidateDevice(InputDevice inputDevice)
    {
        if ((inputDevice.characteristics & InputDeviceCharacteristics.Right) != 0)
        {
            RightController = inputDevice;
            //Debug.Log("Right: Initialized");
        }
        if ((inputDevice.characteristics & InputDeviceCharacteristics.Left) != 0)
        {
            LeftController = inputDevice;
            //Debug.Log("Left: Initialized");
        }
        if ((inputDevice.characteristics & InputDeviceCharacteristics.HeadMounted) != 0)
        {
            Hmd = inputDevice;
            //Debug.Log("HeadMounted: Initialized");
        }
    }

    private void InitializeInputDevices()
    {
        List<InputDevice> devices = new List<InputDevice>();
        InputDevices.GetDevices(devices);
    
        foreach (var item in devices)
        {
            Debug.Log(item.name  + item.characteristics );
            ValidateDevice(item);
        }

    }

}
