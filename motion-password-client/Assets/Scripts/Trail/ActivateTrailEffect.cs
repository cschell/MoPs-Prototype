using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.XR;
using InputDevice = UnityEngine.XR.InputDevice;

public class ActivateTrailEffect : MonoBehaviour
{
    [SerializeField] private bool rightHand = true;
    private TrailRenderer trailRendererObject;
    private bool trailIsActivated = false;
    private InputDevice handDevice;
    private bool didPressGrip;
    private bool didPressPrimary;

    // Start is called before the first frame update
    private void Start()
    {
        var trailRenderer = GetComponentInChildren<TrailRenderer>();
        if (trailRenderer != null)
        {
            trailRenderer.emitting = false;
        }

        trailRendererObject = trailRenderer;
        RegisterDevices();
        InputSystem.onDeviceChange += HandleInputDeviceChanged;
    }

    private void HandleInputDeviceChanged(UnityEngine.InputSystem.InputDevice device, InputDeviceChange change)
    {
        var changeOptions = new List<InputDeviceChange>()
        {
            InputDeviceChange.Added,
            InputDeviceChange.Reconnected,
            InputDeviceChange.Enabled,
            InputDeviceChange.UsageChanged,
            InputDeviceChange.ConfigurationChanged
        }; 
        if (changeOptions.Contains(change))
            RegisterDevices();
    }

    private void RegisterDevices()
    {
        const InputDeviceCharacteristics characteristics = InputDeviceCharacteristics.Controller |
                                                           InputDeviceCharacteristics.HeldInHand;
        var handDevices = new List<InputDevice>();
        var handCharacteristics = rightHand ? InputDeviceCharacteristics.Right : InputDeviceCharacteristics.Left;
        InputDevices.GetDevicesWithCharacteristics(characteristics | handCharacteristics, handDevices);
        foreach (var device in handDevices)
        {
            handDevice = device;
        }
    }

    // Update is called once per frame
    private void Update()
    {
        if (!handDevice.isValid)
            return;

        HandleTrailActivation();

        if (trailIsActivated)
            HandleTrailEmitting();
    }

    private void HandleTrailActivation()
    {
        if (!handDevice.TryGetFeatureValue(UnityEngine.XR.CommonUsages.primaryButton, out var primaryPressed))
            return;
        if (didPressPrimary && !primaryPressed)
        {
            trailIsActivated = !trailIsActivated;
            trailRendererObject.gameObject.SetActive(trailIsActivated);
            Debug.Log("Trail mode was " + (trailIsActivated ? "activated" : "deactivated"));
        }

        didPressPrimary = primaryPressed;
    }

    private void HandleTrailEmitting()
    {
        if (!handDevice.TryGetFeatureValue(UnityEngine.XR.CommonUsages.gripButton, out var gripPressed))
            return;

        if (!didPressGrip && gripPressed)
        {
            Debug.Log("Activated trail");
            trailRendererObject.emitting = true;
        }
        else if (didPressGrip && !gripPressed)
        {
            trailRendererObject.emitting = false;
            Debug.Log("Deactivated trail");
        }

        didPressGrip = gripPressed;
    }
}
