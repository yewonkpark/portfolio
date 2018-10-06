
/*CREATE TYPE*/



CREATE TYPE AirlineType AS OBJECT (
    AirCode VARCHAR2(8),
    Name VARCHAR2(20),
    Address VARCHAR2(40)
);

/*1:N relation : within the object where 1 side*/
CREATE TYPE FlightsType AS OBJECT(
    FlightCode VARCHAR2(10),
    Destination VARCHAR2(20),
    Origin VARCHAR2(20),
    Stops VARCHAR2(20), /*check*/
    ArrTime VARCHAR2(5),
    DepTime VARCHAR2(5),
    DayWeek VARCHAR2(8),
    Offers REF AirlineType  /*1:N relation : within the object where 1 side (NOT N side)*/ 
 
);

CREATE TYPE MileProgramType AS OBJECT(
    MileCode VARCHAR2(15),
    Description VARCHAR2(30),
    StartDate VARCHAR2(7)
);

/*M:N relation : make SEPARATE type and name*/
CREATE TYPE GivesType AS OBJECT(
    AirlineGive REF AirlineType,
    MilePgmGive REF MileProgramType
);

CREATE TYPE MenuType AS OBJECT(
    MenuCode VARCHAR2(20),
    Description VARCHAR2(40)
) NOT FINAL; /*do not forget NOT FINAL*/

CREATE TYPE VegetarianType UNDER MenuType (
    ProteinLevel NUMBER(3,2)
);

CREATE TYPE LowfatType UNDER MenuType (
    Kcal NUMBER(5,2)
);

CREATE TYPE KosherType UNDER MenuType (
    KosherOrganization VARCHAR(20)
);

CREATE TYPE PassengersType AS OBJECT(
    PassCode VARCHAR2(15),
    Address VARCHAR2(40),
    Sex VARCHAR2(6),
    Age NUMBER,
    Name VARCHAR(20),
    Subcribe REF MileProgramType /*name*/
);

CREATE TYPE MilesOrderType AS OBJECT(
    OrderCode VARCHAR2(20), 
    Quantity NUMBER,
    ODate VARCHAR2(7)
);


CREATE TYPE HaveType AS OBJECT( /*within passenger-flight relationship*/
    Crew VARCHAR2(200), 
    PilotName VARCHAR2(20),
    HDate VARCHAR2(7),
    Seat VARCHAR2(7),
    FlightsHave REF FlightsType,
    PassengersHave REF PassengersType,
    MileOrderHave REF MileOrderType,
    MenuServe REF MenuType
);


/*CREATE TABLE*/

CREATE TABLE Airline OF AirlineType (
    AirCode PRIMARY KEY
);

CREATE TABLE Flights OF FlightsType (
    FlightCode PRIMARY KEY,
    SCOPE FOR (Offers) IS Airline
);

CREATE TABLE MileProgram OF MileProgramType (
    MileCode PRIMARY KEY
);
/*M:N relation : no PRIMAY KEY*/
CREATE TABLE Gives OF GivesType (
    SCOPE FOR (AirlineGive) IS Airline,
    SCOPE FOR (MilePgmGive) IS MileProgram
);

CREATE TABLE Menu OF MenuType(
    MenuCode PRIMARY KEY
);

CREATE TABLE Vegetarian OF VegetarianType;
CREATE TABLE Lowfat OF LowfatType;
CREATE TABLE Koshern OF KosherType;

CREATE TABLE Passengers OF PassengersType (
    PassCode PRIMARY KEY,
    SCOPE FOR (Subcribe) IS MileProgram
);

CREATE TABLE MilesOrder OF MilesOrderType (
    OrderCode PRIMARY KEY
);

CREATE TABLE Have OF HaveType (
    SCOPE FOR (FlightsHave) IS Flights,
    SCOPE FOR (PassengersHave) IS Passengers,
    SCOPE FOR (MileOrderHave) IS MileOrder,
    SCOPE FOR (MenuServe) IS  Menu
);



/*a) List the airline name that passenger John Wood has taken on the 04/APR/2012. (20 %)*/
SELECT DEFER(H.FlightHave).Offers.Name
FROM Have H
WHERE DEREF(H.PassengerHave).Name = 'John Wood' and H.Date = '04APR2012';

SELECT H.FlightHave.Offers.Name
FROM Have H
WHERE H.PassengerHave.Name = 'John Wood' and H.Date = '04APR2012';

/*b) List the description of the mile programme of passengers that live in London (20 %)*/
SELECT DEREF(P.Subscribe).Description
FROM Passenger P
WHERE P.Address LIKE %London%;

SELECT P.Subscribe.Description
FROM Passenger P
WHERE P.Address LIKE %London%;
