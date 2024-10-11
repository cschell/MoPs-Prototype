using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.XR;


public class TrailHelper
{
    public static InputDevice RegisterDevices(bool rightHand)
    {
        const InputDeviceCharacteristics characteristics = InputDeviceCharacteristics.Controller |
                                                           InputDeviceCharacteristics.HeldInHand;
        var handDevices = new List<InputDevice>();
        var handCharacteristics = rightHand ? InputDeviceCharacteristics.Right : InputDeviceCharacteristics.Left;
        InputDevices.GetDevicesWithCharacteristics(characteristics | handCharacteristics, handDevices);
        foreach (var device in handDevices)
        {
            return device;
        }

        return new InputDevice();
    }
    
    public static InputDevice GetPrimaryHandDevice(Hand hand)
    {
        const InputDeviceCharacteristics controllerCharacteristics = InputDeviceCharacteristics.Controller |
                                                                     InputDeviceCharacteristics.HeldInHand;
        
        var handDevices = new List<InputDevice>();

        InputDevices.GetDevicesWithCharacteristics(controllerCharacteristics | GetHandCharacteristic(hand), handDevices);
        
        return handDevices.Count > 0 ?
            // Return the first device (assuming primary)
            handDevices[0] :
            // No device found, return a default InputDevice
            new InputDevice();
    }

    private static InputDeviceCharacteristics GetHandCharacteristic(Hand hand)
    {
        switch (hand)
        {
            case Hand.Left:
                return InputDeviceCharacteristics.Left;
            case Hand.Right:
                return InputDeviceCharacteristics.Right;
            default:
                Debug.LogError($"Invalid hand specified: {hand}");
                throw new ArgumentOutOfRangeException(nameof(hand), hand, "Invalid hand specified.");
        }
    }
    
    public enum Hand
    {
        Left,
        Right
    }

    public static void HandleTrailEmitting(InputDevice currentHandDevice, TrailRenderer currentTrailRenderer)
    {
        if (!currentHandDevice.TryGetFeatureValue(CommonUsages.gripButton, out var gripPressed)) return;

        currentTrailRenderer.emitting = gripPressed;

        currentTrailRenderer.time = gripPressed ? 1f : 200f;
    }
    

    public static void HandleTrailEmittingV2(InputDevice handDevice, TrailRenderer trailRenderer)
    {
        if (!handDevice.TryGetFeatureValue(CommonUsages.gripButton, out var gripPressed)) return;

        trailRenderer.emitting = gripPressed;

        float trailLength;

        if (gripPressed)
        {
            trailLength = 1f; // Short trail
        }
        else
        {
            trailLength = 5f; // Long trail
        }

        trailRenderer.time = trailLength;
    }

}
