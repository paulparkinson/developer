# Develop with Oracle Database 23ai and GraalVM using Spring Boot

## Introduction

This lab walks you through the steps to develop with Oracle Database 23ai and GraalVM using Spring Boot
Estimated Time: 30 minutes

### Objectives

In this lab, you will:
- Develop with Oracle Database 23ai and GraalVM using Spring Boot

### Prerequisites

This lab assumes you have:
- Provisioned environment with Java GraalVM and Maven.


## Task 1: Clone and Build

   1. Clone the repository:

      git clone https://github.com/paulparkinson/react-graphql-springboot-jpa-oracle-ucp-example.git

   2. See https://blogs.oracle.com/developers/post/develop-react-graphql-spring-boot-data-jpa-ucp-oracle
3. cd spring-data-jpa-graphql-ucp-oracle
      Modify src/main/resources/application.properties to set values for spring.datasource.url, spring.datasource.username, and spring.datasource.password
      Run mvn clean install
      Run java -jar target/spring-data-jpa-graphql-oracle-0.0.1-SNAPSHOT.jar
      (In a separate terminal/console) cd react-graphql
      Run yarn add @apollo/client graphql (this is only necessary once for the project)
      Run npm run build
      Run npm start
      A browser window should open to http://localhost:3000/ which is a React app that will use Apollo to make a GraphQL query against a Spring Boot service running on localhost:8080 which in turn uses JPA to query an Oracle database via a connection obtained from UCP.

   3. `cd ...` and ...

       ```
       <copy>
      asdf
       </copy>
       ```
     
      The ....

   4. Depending on the compute shape you want to use, modify `variables.tf` (instance_shape variable) and setup.sh (parallel_gpu_count). If you have a cluster of n GPUs, the GPU count should also be n.



## Acknowledgements
* **Author** - Paul Parkinson, Architect and Developer Advocate
* **Last Updated By/Date** - Paul Parkinson, 2024
