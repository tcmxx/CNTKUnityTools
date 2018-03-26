using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityCNTK;
using CNTK;
using UnityCNTK.ReinforcementLearning;


public class PongEnvironment : MonoBehaviour,IRLEnvironment {

    public ControlSource playerLeftControl;
    public ControlSource playerRightControl; 

    public float failureReward = -1;
    public float winReward = 1;
    public float hitBallReward = 0.1f;

    public float racketSpeed = 0.02f;
    public float ballSpeed = 0.01f;
    public float racketWidth = 0.05f;

    public readonly int ActionUp = 2;
    public readonly int ActionDown = 0;
    public readonly int ActionStay = 1;

    public float leftStartX = -1;
    public float rightStartX = 1;
    public Vector2 arenaSize = new Vector2(2.2f, 1.0f);


    [Header("Informations")]
    public float leftHitOrMiss = 0;
    public float rightHitOrMiss = 0;

    public GameState CurrentGameState { get { return currentGameState; } }
    private GameState currentGameState;
    


    protected int step = 0;
    public int framesPerStep = 5;

    public struct GameState
    {
        public Vector2 ballVelocity;
        public Vector2 ballPosition;
        public float leftY;
        public float rightY;
        public int gameWinPlayer;
        public float rewardLastStepLeft;
        public float rewardLastStepRight;
    }

    public enum ControlSource
    {
        FromStep,
        FromPlayerInput,
        SimpleAI
    }

    public int GameWinPlayer
    {
        get
        {
            return currentGameState.gameWinPlayer;
        }
        protected set
        {
            currentGameState.gameWinPlayer = value;
        }
    }



    private void Start()
    {
        Physics.autoSimulation = false;
        Reset();
    }


    public float[] CurrentState(int actor=0)
    {
        float[] result = null;
        if (actor == 0)
        {
            result = new float[] {
                currentGameState.leftY,
                currentGameState.rightY,
                currentGameState.ballPosition.x,
                currentGameState.ballPosition.y,
                currentGameState.ballVelocity.x,
                currentGameState.ballVelocity.y
            };
        }
        else
        {
            result = new float[] {
                currentGameState.rightY,
                currentGameState.leftY,
                -currentGameState.ballPosition.x,
                currentGameState.ballPosition.y,
                -currentGameState.ballVelocity.x,
                currentGameState.ballVelocity.y
            };
        }
        return result;
    }
    /// <summary>
    /// steps from the start of this episode
    /// </summary>
    /// <returns></returns>
    public int CurrentStep()
    {
        return step;
    }

    public bool IsResolved()
    {
        return true;
    }

    //public void Step(float[] actions) { Debug.LogError("Need two actor"); }

    /// <summary>
    /// Actions: 0:down, 1 not move, 2 up
    /// </summary>
    /// <param name="actions"></param>
    public void Step(params float[][] actions)
    {
        //get action from different sources
        int actionLeft;
        if (playerLeftControl == ControlSource.FromPlayerInput) {
            actionLeft = Input.GetButton("Up")?2:(Input.GetButton("Down")?0:1);
        } else if (playerLeftControl == ControlSource.SimpleAI) {
            actionLeft = SimpleAI(0);
        } else {
            actionLeft = (int)actions[0][0]; 
        }

        int actionRight;
        if (playerRightControl == ControlSource.FromPlayerInput)
        {
            actionRight = Input.GetButton("Up") ? 2 : (Input.GetButton("Down") ? 0 : 1);
        }
        else if (playerRightControl == ControlSource.SimpleAI)
        {
            actionRight = SimpleAI(1);
        }
        else
        {
            actionRight = (int)actions[playerLeftControl == ControlSource.FromStep?1:0][0];
        }

        //clear the reward 
        currentGameState.rewardLastStepLeft = 0;
        currentGameState.rewardLastStepRight = 0;
        for (int i = 0; i < framesPerStep; ++i)
        {


            Debug.Assert(actionLeft >= ActionDown && actionLeft < ActionUp + 1);
            Debug.Assert(actionLeft >= ActionDown && actionLeft < ActionUp + 1);

            //move the rackets
            currentGameState.leftY += racketSpeed * (actionLeft - 1);
            currentGameState.leftY = Mathf.Clamp(currentGameState.leftY, -arenaSize.y / 2 + racketWidth / 2, arenaSize.y / 2 - racketWidth / 2);
            currentGameState.rightY += racketSpeed * (actionRight - 1);
            currentGameState.rightY = Mathf.Clamp(currentGameState.rightY, -arenaSize.y / 2 + racketWidth / 2, arenaSize.y / 2 - racketWidth / 2);

            //move the ball
            Vector2 oldBallPosition = currentGameState.ballPosition;
            currentGameState.ballPosition += currentGameState.ballVelocity;

            //detect collision of ball with wall
            Vector2 newBallVel = currentGameState.ballVelocity;
            if (currentGameState.ballPosition.y > arenaSize.y / 2 || currentGameState.ballPosition.y < -arenaSize.y / 2)
            {
                newBallVel.y = -newBallVel.y;

            }
            if (currentGameState.ballPosition.x > arenaSize.x / 2)
            {
                currentGameState.rewardLastStepLeft += winReward;
                currentGameState.rewardLastStepRight += failureReward;
                GameWinPlayer = 0;
                break;
            }
            else if (currentGameState.ballPosition.x < -arenaSize.x / 2)
            {
                currentGameState.rewardLastStepRight += winReward;
                currentGameState.rewardLastStepLeft += failureReward;
                GameWinPlayer = 1;
                break;
            }

            //detect collision of the ball with the rackets
            if (currentGameState.ballPosition.x < leftStartX && oldBallPosition.x > leftStartX)
            {
                Vector2 moveVector = (currentGameState.ballPosition - oldBallPosition);
                float yHit = (moveVector * Mathf.Abs((oldBallPosition.x - leftStartX) / moveVector.x) + oldBallPosition).y;
                float yHitRatio = (currentGameState.leftY - yHit) / (racketWidth / 2);
                if (Mathf.Abs(yHitRatio) < 1)
                {
                    //hit the left racket
                    newBallVel.x = -newBallVel.x;
                    newBallVel.y = -Mathf.Abs(newBallVel.x) * yHitRatio * 2;
                    newBallVel = newBallVel.normalized * ballSpeed;
                    currentGameState.rewardLastStepLeft += hitBallReward;
                    leftHitOrMiss = 1;
                }
                else
                {
                    leftHitOrMiss = -1;
                }
            }
            else if (currentGameState.ballPosition.x > rightStartX && oldBallPosition.x < rightStartX)
            {
                Vector2 moveVector = (currentGameState.ballPosition - oldBallPosition);
                float yHit = (moveVector * Mathf.Abs((oldBallPosition.x - rightStartX) / moveVector.x) + oldBallPosition).y;
                float yHitRatio = (currentGameState.rightY - yHit) / (racketWidth / 2);
                if (Mathf.Abs(yHitRatio) < 1)
                {
                    //hit the right racket
                    newBallVel.x = -newBallVel.x;
                    newBallVel.y = -Mathf.Abs(newBallVel.x) * yHitRatio * 2;
                    newBallVel = newBallVel.normalized * ballSpeed;
                    currentGameState.rewardLastStepRight += hitBallReward;
                    rightHitOrMiss = 1;
                }
                else
                {
                    rightHitOrMiss = -1;
                }
            }
            else
            {
                leftHitOrMiss = 0;
                rightHitOrMiss = 0;
            }

            //update the velocity
            currentGameState.ballVelocity = newBallVel;
        }

        //test
        //currentGameState.rewardLastStepLeft = (0.2f - Mathf.Abs(currentGameState.leftY - currentGameState.ballPosition.y))*5;
        //currentGameState.rewardLastStepRight = (0.2f - Mathf.Abs(currentGameState.rightY - currentGameState.ballPosition.y)) * 5;
        //if(Mathf.Abs(currentGameState.leftY - currentGameState.ballPosition.y) < racketWidth / 2)
        //{
        //    currentGameState.rewardLastStepLeft = 0.1f;
        //}
        //else
        //{
        //    currentGameState.rewardLastStepLeft = 0.0f;
       // }
        //currentGameState.rewardLastStepLeft = 0.02f;

        step++;
    }


    public float LastReward(int actor = 0)
    {
        return actor == 0 ? currentGameState.rewardLastStepLeft : currentGameState.rewardLastStepRight;
    }

    public bool IsEnd()
    {
        return currentGameState.gameWinPlayer >= 0;
    }

    // to be implemented by the developer
    public void Reset()
    {
        currentGameState.leftY = 0;
        currentGameState.rightY = 0;
        currentGameState.ballPosition = Vector2.zero;
        Vector2 initialVel = Random.insideUnitCircle;
        if (Mathf.Abs(initialVel.y) > Mathf.Abs(initialVel.x))
        {
            float temp = initialVel.y;
            initialVel.y = initialVel.x;
            initialVel.x = temp;
        }
        currentGameState.ballVelocity = initialVel.normalized * ballSpeed;
        currentGameState.rewardLastStepRight = 0;
        currentGameState.rewardLastStepLeft = 0;
        currentGameState.gameWinPlayer = -1;
        step = 0;
    }

    protected int SimpleAI(int actorNum)
    {
        var states = CurrentState(actorNum);
        return states[0] > states[3] ? ActionDown : ActionUp;
    }
}
