import model
import simulationUtils

if __name__ == "__main__":
    # model.run(simulationUtils.initSimulation,
    #     learning_rate=0.001,
    #     gamma=0.98,
    #     buffer_max_size=80000,
    #     batch_size=10,
    #     target_update_interval=10,
    #     replay_buffer_start_size=1000)

    tf = simulationUtils.initSimulation()
    for _ in range(simulationUtils.SIM_TIME // simulationUtils.STEP_LENGTH):
        tf.step(0)
    tf.close()