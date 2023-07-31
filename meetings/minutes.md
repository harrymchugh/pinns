# Meeting notes

Regular meetings between Harry McHugh and Adrian Jackson were held throughout the dissertation period.

A summary of key decisions and notes are provided in this file, some meetings have been excluded as they were simply catch-up meetings with no material progress reported or discussed.

Meetings began in earnest in January 2023 after Adrian reccomended we conduct a literature to determine the desired scope of the project.

## 13/01/2023
We discussed the outcome of the literature, particularly focussing on PINNs and how we may explore them in context of CFD.

This included whether we would focus on integrating PINNs into OpenFOAM or simply exploring them in contrast to OpenFOAM.

Harry had a preference to viewing them in contast to OpenFOAM as the literature suggested they are capable of fully learining the underlying physics.

Therefore we could assess both the accuracy and the performance of PINNs in respect to OpenFOAM.

## 23/02/2023
Harry explained how he had been investigating DeepONets as a possible alternative to PINNs given the success that had been recorded in the literature.

Adrian reccomended simplifying the scope and focussing on a simple Python application rather than diving straight into a PINN that replicates OpenFOAM.

## 20/04/2023
Following the feedback from the previous meeting Harry showed the progress he had made using PyTorch to create a PINN that implements the 1D heat diffusion equation. 

Adrian agreed this was a good stepping stone approach to bridge the gap to OpenFOAM.

Harry and Adrian agreed that the next logical step was to extend the PINN to predict 2D fluid properties. 

Harry requested a meeting with Angus Creech to confirm he was on the right track with the Navier-Stokes theorey that would be required to embed the Navier-Stokes PDE into a fluid PINN.

## 05/05/2023
Harry, Angus and Adrian met to discuss the embedding of the Navier-Stokes equation into the PINN framework Harry had extended from the 1D diffusion example. 

Angus encouraged Harry to use a laminar fluid flow case from OpenFOAM such as the icoFOAM implementation of the lid-driven cavity, as this was a well studied case (it is provided as an OpenFOAM tutorial) and we could have confidnece in the results.

## 19/05/2023 
Harry showed Adrian the implementation of the lid-driven cavity PINN using OpenFOAM as the input.

This result was the outcome of the meeting with Angus and the first successful PINN capable of producing fluid properties.

In this meeting Adrian and Harry discussed different experiments that could be used to assess the adaptability of PINNs in contrast to OpenFOAM, including varying geometry and mesh sizes.

## 01/06/2023
Harry was going away on holiday and plans to spend a signficant amount of time refactoring the CFDPINN codebase into a proper Python applciation rather than leaving it in its current notebook status.

Adrian and Harry agreed to have a catch-up meeting on Harry's return from holiday.

## 23/06/2023
This meeting was a catch-up after Harry's holiday where Harry showcased the experiments run after refactoring that were proposed during the meeting held 19/05.

Harry showed how the PINN was able to adapt to mesh-size variations but could not handle change in geometry and shape.

We also held a discussion about potentially benchmarking performance on a number of different GPUs should there be enough time towards the end of the project.

## 06/07/2023
This meeting was used as an opportunity to review Harry's first draft of the dissertation report including section layout and content. 

Adrian reccomended being more explicity about the goals of the project early in the report so they can be referenced during the conclusion phase.
