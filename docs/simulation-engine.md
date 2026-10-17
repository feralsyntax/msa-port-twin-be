# Simulation Engine — Initial Design

## Purpose

Simulate vessel arrivals, waiting queues, berth allocation, and service completion to help demonstrate port congestion and operational decision-making.

## Inputs

- Simulation duration in hours
- Expected number of vessel arrivals
- Interval between arrivals
- Average berth service time
- Available berths
- Vessel length and draft
- Berth maximum vessel length and draft

## Processing

1. Generate deterministic vessel arrival times.
2. Identify berths compatible with each vessel.
3. Assign the vessel to the compatible berth that becomes available soonest.
4. Queue the vessel when no compatible berth is available immediately.
5. Calculate waiting time, service duration, departure time, and turnaround time.
6. Aggregate simulation outcomes into a summary.

## Outputs

### Vessel-level results

- Simulated vessel identifier
- Arrival time
- Assigned berth, if any
- Service start time, if service starts within the simulation horizon
- Departure time, if service completes within the simulation horizon
- Waiting time, when calculable
- Vessel status at the end of the run

### Simulation summary

- Total scheduled arrivals
- Vessels served
- Vessels waiting at the end
- Vessels still in service at the end
- Average waiting time for vessels that started service
- Average turnaround time for vessels that completed service
- Berth utilization during the simulation horizon

## Initial assumptions

- The simulation uses deterministic arrival intervals.
- Service duration is constant for the initial version.
- Each berth serves one vessel at a time.
- Vessel length and draft determine berth compatibility.
- Berths are assumed available at simulation hour zero unless later configuration explicitly states otherwise.
- Results are simulated estimates, not live port measurements.
- The engine is independent of Django and database queries.
- AI predictions and hydrodynamic calculations are outside the initial queueing engine.

## Architecture

The simulation engine performs calculations independently of Django views and models. Django will later load scenario data, call the engine, and persist the resulting run and result records.
