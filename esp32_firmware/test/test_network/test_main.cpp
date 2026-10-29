#include <unity.h>
#include "network.h" 

void setUp(void) {}
void tearDown(void) {}

void test_network_output(void) {
    float obs[4] = {0.0, 1.2, -0.5, 3.1}; 
    
    float action = predict_action(obs); 
    
    TEST_ASSERT_TRUE(action >= -20.0 && action <= 20.0);
}

int main(int argc, char **argv) {
    UNITY_BEGIN();
    RUN_TEST(test_network_output);
    return UNITY_END();
}