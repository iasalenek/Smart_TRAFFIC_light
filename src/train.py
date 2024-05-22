import model
import simulationUtils
import numpy as np

if __name__ == "__main__":
    # model.run(simulationUtils.initSimulation,
    #     learning_rate=0.001,
    #     gamma=0.98,
    #     buffer_max_size=80000,
    #     batch_size=10,
    #     target_update_interval=10,
    #     replay_buffer_start_size=1000)

    tf = simulationUtils.initSimulation()
    for _ in range(10):
        
        indexes = np.nonzero(tf.get_avail_agent_actions(0))[0]
        print(len(tf.get_avail_agent_actions(0)))
        reward, term = tf.step((np.nonzero(tf.get_avail_agent_actions(0))[0][0],
                       np.nonzero(tf.get_avail_agent_actions(1))[0][0],
                       np.nonzero(tf.get_avail_agent_actions(2))[0][0]))
        if term:
            break
    tf.close()

    tf = simulationUtils.initSimulation()
    for _ in range(100000):
        
        indexes = np.nonzero(tf.get_avail_agent_actions(0))[0]
        reward, term = tf.step((np.nonzero(tf.get_avail_agent_actions(0))[0][0],
                       np.nonzero(tf.get_avail_agent_actions(1))[0][0],
                       np.nonzero(tf.get_avail_agent_actions(2))[0][0]))
        if term:
            break
    tf.close()