# msa-port-twin-be
API powering an AI-powered maritime digital twin for port monitoring, vessel simulation, berth scheduling, and scenario analysis.

# Set up Instructions
Install requirements
- Development mode
    ```bash
    pip install -r requirements-dev.txt
    ```
- Production mode
    ```bash
    pip install -r requirements.txt
    ```

# Demonstration Data

MSA Port Twin provides synthetic demonstration data for local development and application demonstrations.

## Load demonstration data

```bash
python manage.py seed_demo_data
```

# Tests

The `port_twin_app` has test modules for:
- The API
- The simulation engine
    - Simulations
    - Simulation API
    - Simulation detail
    - Simulation execution
- Operations overview API

## Run Tests

To run all tests, use the following command:

```bash
python manage.py test
```

## Run A Single Test

To run a single, specific test, use the following command

```bash
python manage.py test port_twin_app.tests.[the_name_of_the_test_file]
```
Replace [the_name_of_the_test_file] with the actual test file you want. For example, to test simulations in the test file `test_simulations`, use the command:

```bash
python manage.py test port_twin_app.tests.test_simulations
```