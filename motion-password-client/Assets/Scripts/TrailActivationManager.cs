using UnityEngine;
using UnityEngine.XR;

public class TrailActivationManager : MonoBehaviour
{
    [SerializeField] private GameObject leftTrail;
    [SerializeField] private GameObject rightTrail;
    private InputDevice _rightHandDevice;
    private InputDevice _leftHandDevice;
    private TrailRenderer _leftTrailRenderer;
    private TrailRenderer _rightTrailRenderer;
    private static int _emittedPositionCounts;
    private static int _nonEmittedPositionCounts;
    private bool _trailIsActive;

    private void Start()
    {
        _leftTrailRenderer = leftTrail.GetComponent<TrailRenderer>();
        _rightTrailRenderer = rightTrail.GetComponent<TrailRenderer>();
        _leftTrailRenderer.emitting = false;
        _rightTrailRenderer.emitting = false;
        leftTrail.SetActive(true);
        rightTrail.SetActive(false);
        _rightHandDevice = TrailHelper.GetPrimaryHandDevice(TrailHelper.Hand.Right);
        _leftHandDevice = TrailHelper.GetPrimaryHandDevice(TrailHelper.Hand.Left);
    }

    private void Update()
    {
        
        _trailIsActive = true;
        rightTrail.SetActive(true);
        
        if (!_rightHandDevice.isValid | !_leftHandDevice.isValid)
        {
            _rightHandDevice = TrailHelper.GetPrimaryHandDevice(TrailHelper.Hand.Right);
            _leftHandDevice = TrailHelper.GetPrimaryHandDevice(TrailHelper.Hand.Left);
            return;
        }

        HandleButtonAccessibility(_leftTrailRenderer);
        HandleButtonAccessibility(_rightTrailRenderer);
        if (!_trailIsActive) return;
        TrailHelper.HandleTrailEmittingV2(_rightHandDevice, _rightTrailRenderer);
        TrailHelper.HandleTrailEmittingV2(_leftHandDevice, _leftTrailRenderer);
    }


    private void HandleButtonAccessibility(TrailRenderer trailRenderer)
    {
        if (trailRenderer.positionCount < _emittedPositionCounts + _nonEmittedPositionCounts)
        {
            _emittedPositionCounts = 0;
            _nonEmittedPositionCounts = 0;
        }

        var diff = trailRenderer.positionCount - (_emittedPositionCounts + _nonEmittedPositionCounts);
        if (trailRenderer.emitting)
        {
            _emittedPositionCounts += diff;
        }
        else
        {
            _nonEmittedPositionCounts += diff;
        }
    }

}

