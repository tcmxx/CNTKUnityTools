using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityCNTK;

public class PoleGameEnviroment : MonoBehaviour, IRLEnvironment
{

    public bool isContinuou = true;
    public float deltaTime = 0.02f;
    public float damp = 0.01f;
    public int framesPerStep = 5;
    public float gravity = 1;
    protected float reward = 0;
    protected bool done;
    protected int step = 0;


    public float velR;
    public float angleR;    //in radian

    public GameObject poleObjectRef;

    private void Start()
    {
        Physics.autoSimulation = false;
        Reset();
    }


    private void Update()
    {
        poleObjectRef.transform.rotation = Quaternion.Euler(0, 0, angleR * Mathf.Rad2Deg + 180);
    }

    public float[] CurrentState(int actor = 0)
    {
            var state = new float[2];
            state[0] = velR;
            state[1] = angleR;
            return state;
    }
    public int CurrentStep()
    {
        return step;
    }

    public bool IsResolved()
    {
        return true;
    }

    public void Step(params float[][] act)
    {
        Step(act[0]);
    }
    public void Step(float[] act)
    {
        reward = 0;
        for (int i = 0; i < framesPerStep; ++i)
        {
            float gavityTorque = Mathf.Sin(angleR);
            float torque = 0;
            float dampeTorque = -velR * velR * damp*Mathf.Sign(velR);
            if (isContinuou)
            {
                torque = act[0];
                torque = Mathf.Clamp(torque, -3, 3);
            }
            else
            {
                torque = (act[0] ==0 ?-3.0f:3.0f);
            }
            velR += deltaTime * (torque + gavityTorque + dampeTorque);
            angleR += deltaTime * velR;

            if (angleR < -Mathf.PI) {
                angleR = angleR + 2 * Mathf.PI;
            }else if(angleR > Mathf.PI)
            {
                angleR = angleR - 2 * Mathf.PI;
            }

            reward = -Mathf.Abs(angleR)+Mathf.PI/2 - Mathf.Abs(velR);
            reward /= framesPerStep*10;
        }
        
        step++;
    }


    public float LastReward(int actor = 0)
    {
        return reward;
    }

    public bool IsEnd()
    {
        return done;
    }

    // to be implemented by the developer
    public void Reset()
    {
        angleR = Random.Range(-Mathf.PI/2, Mathf.PI/2);
        velR = Random.Range(0, 0);
        done = false;
        reward = 0;
        step = 0;
    }
}
