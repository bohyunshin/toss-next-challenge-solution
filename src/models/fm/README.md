# Factorization Machine (FM) Models

This directory contains implementations of various Factorization Machine models and their extensions.

## Class Inheritance Structure

```mermaid
graph LR
    Base --> LogisticRegression
    Base --> FMBase
    Base --> F16lre
    Base --> FFMBase
    Base --> DCNBase
    Base --> DCNv2Base
    
    FMBase --> FM
    FMBase --> SequenceFM
    
    DeepFMBase --> DeepFM
    DeepFMBase --> SequenceDeepFM
    
    FFMBase --> FFM
    FFMBase --> SequenceFFM
    
    xDeepFMBase --> xDeepFM
    xDeepFMBase --> SequencexDeepFM
    
    DCNBase --> DCN
    DCNBase --> SequenceDCN
    
    DCNv2Base --> DCNv2
    DCNv2Base --> SequenceDCNv2
```
