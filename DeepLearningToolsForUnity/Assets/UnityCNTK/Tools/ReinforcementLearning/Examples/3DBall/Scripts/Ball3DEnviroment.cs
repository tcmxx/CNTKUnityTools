using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityCNTK;

public class Ball3DEnviroment : MonoBehaviour, IRLEnvironment
{
    [Header("Specific to Ball3D")]
    public GameObject ball;
    public bool isContinuou = true;

    public int framesPerStep = 5;
    public bool is3D = false;
    protected float reward = 0;
    protected bool done;

    protected int step = 0;

    private void Start()
    {
        Physics.autoSimulation = false;
    }


    public float[] CurrentState()
    {
        if (is3D)
        {
            var state = new float[8];
            state[0] = (gameObject.transform.rotation.z);
            state[1] = (gameObject.transform.rotation.x);
            state[2] = ((ball.transform.position.x - gameObject.transform.position.x));
            state[3] = ((ball.transform.position.y - gameObject.transform.position.y));
            state[4] = ((ball.transform.position.z - gameObject.transform.position.z));
            state[5] = (ball.transform.GetComponent<Rigidbody>().velocity.x);
            state[6] = (ball.transform.GetComponent<Rigidbody>().velocity.y);
            state[7] = (ball.transform.GetComponent<Rigidbody>().velocity.z);
            return state;
        }
        else
        {
            var state = new float[5];
            state[0] = (gameObject.transform.rotation.z)*4;
            //state[1] = (gameObject.transform.rotation.x);
            state[1] = ((ball.transform.position.x - gameObject.transform.position.x));
            state[2] = ((ball.transform.position.y - gameObject.transform.position.y));
            //state[4] = ((ball.transform.position.z - gameObject.transform.position.z));
            state[3] = (ball.transform.GetComponent<Rigidbody>().velocity.x);
            state[4] = (ball.transform.GetComponent<Rigidbody>().velocity.y);
            //state[7] = (ball.transform.GetComponent<Rigidbody>().velocity.z);
            return state;
        }
    }
    public int CurrentStep()
    {
        return step;
    }

    public bool IsResolved()
    {
        return true;
    }


    public void Step(float[] act)
    {
        reward = 0;
        for (int i = 0; i < framesPerStep; ++i)
        {
            if (isContinuou)
            {
                float action_z = act[0];

                if (action_z > 2f)
                {
                    action_z = 2f;
                }
                if (action_z < -2f)
                {
                    action_z = -2f;
                }
                if ((gameObject.transform.rotation.z < 0.25f && action_z > 0f) ||
                    (gameObject.transform.rotation.z > -0.25f && action_z < 0f))
                {
                    gameObject.transform.Rotate(new Vector3(0, 0, 1), action_z);
                }
                if (is3D)
                {
                    float action_x = act[1];
                    if (action_x > 2f)
                    {
                        action_x = 2f;
                    }
                    if (action_x < -2f)
                    {
                        action_x = -2f;
                    }
                    if ((gameObject.transform.rotation.x < 0.25f && action_x > 0f) ||
                        (gameObject.transform.rotation.x > -0.25f && action_x < 0f))
                    {
                        gameObject.transform.Rotate(new Vector3(1, 0, 0), action_x);
                    }
                }
                if (done == false)
                {
                    reward += 0.1f / framesPerStep;
                }
            }
            else
            {
                int action = (int)act[0];
                if (action == 0 || action == 1)
                {
                    action = (action * 2) - 1;
                    float changeValue = action * 2f;
                    if ((gameObject.transform.rotation.z < 0.25f && changeValue > 0f) ||
                        (gameObject.transform.rotation.z > -0.25f && changeValue < 0f))
                    {
                        gameObject.transform.Rotate(new Vector3(0, 0, 1), changeValue);
                    }
                }
                if (action == 2 || action == 3)
                {
                    action = ((action - 2) * 2) - 1;
                    float changeValue = action * 2f;
                    if ((gameObject.transform.rotation.x < 0.25f && changeValue > 0f) ||
                        (gameObject.transform.rotation.x > -0.25f && changeValue < 0f))
                    {
                        gameObject.transform.Rotate(new Vector3(1, 0, 0), changeValue);
                    }
                }
                if (done == false)
                {
                    reward += 0.1f/framesPerStep;
                }
            }
            Physics.Simulate(Time.fixedDeltaTime);
            //reward -= Mathf.Abs(ball.transform.GetComponent<Rigidbody>().velocity.x/50);
        }


        if ((ball.transform.position.y - gameObject.transform.position.y) < -2f ||
            Mathf.Abs(ball.transform.position.x - gameObject.transform.position.x) > 3f ||
            Mathf.Abs(ball.transform.position.z - gameObject.transform.position.z) > 3f)
        {
            done = true;
            reward = -1f;
        }
        //test
        //reward = -Mathf.Abs(gameObject.transform.rotation.z);
        step++;
    }


    public float LastReward()
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
        step = 0;
        reward = 0;
        done = false;
        gameObject.transform.rotation = new Quaternion(0f, 0f, 0f, 0f);
        if (is3D)
        {
            gameObject.transform.Rotate(new Vector3(1, 0, 0), Random.Range(-10f, 10f));
            ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(-1.5f, 1.5f)) + gameObject.transform.position;
            ball.GetComponent<Rigidbody>().constraints = RigidbodyConstraints.None;
        }
        else
        {
            ball.transform.position = new Vector3(Random.Range(-1.5f, 1.5f), 4f, Random.Range(0, 0)) + gameObject.transform.position;
            ball.GetComponent<Rigidbody>().constraints = RigidbodyConstraints.FreezePositionZ;
        }
        //
        gameObject.transform.Rotate(new Vector3(0, 0, 1), Random.Range(-10f, 10f));
        ball.GetComponent<Rigidbody>().velocity = new Vector3(0f, 0f, 0f);
        
    }
}
