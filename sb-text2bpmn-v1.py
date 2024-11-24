from langchain_openai import ChatOpenAI
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import re

load_dotenv()

# Define enums for the types
class TaskType(str, Enum):
    USER = "user"
    SERVICE = "service" 
    MANUAL = "manual"
    SCRIPT = "script"
    BUSINESS_RULE = "business_rule"

class GatewayType(str, Enum):
    EXCLUSIVE = "exclusive"
    PARALLEL = "parallel"
    INCLUSIVE = "inclusive"
    EVENT_BASED = "event_based"

class EventType(str, Enum):
    START = "start"
    INTERMEDIATE = "intermediate"
    END = "end"

class Connection(BaseModel):
    source_id: str
    target_id: str
    condition: Optional[str] = None

class Task(BaseModel):
    id: str
    name: str
    type: TaskType
    lane_id: Optional[str] = None
    incoming: List[str] = Field(default_factory=list)
    outgoing: List[str] = Field(default_factory=list)

class Gateway(BaseModel):
    id: str
    name: str
    type: GatewayType
    lane_id: Optional[str] = None
    incoming: List[str] = Field(default_factory=list)
    outgoing: List[str] = Field(default_factory=list)
    conditions: Dict[str, str] = Field(default_factory=dict)  # outgoing_id: condition

class Event(BaseModel):
    id: str
    name: str
    type: EventType
    lane_id: Optional[str] = None
    incoming: List[str] = Field(default_factory=list)
    outgoing: List[str] = Field(default_factory=list)

class Lane(BaseModel):
    id: str
    name: str
    parent_pool_id: str

class Pool(BaseModel):
    id: str
    name: str
    lanes: List[Lane] = Field(default_factory=list)

class BPMNProcess(BaseModel):
    """Main class representing the entire BPMN process"""
    process_name: str
    pools: List[Pool] = Field(default_factory=list)
    lanes: List[Lane] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    events: List[Event] = Field(default_factory=list)
    gateways: List[Gateway] = Field(default_factory=list)
    connections: List[Connection] = Field(default_factory=list)

class BPMNProcessCorrected(BPMNProcess):
    """Main class representing the entire BPMN process with validation comment"""
    comment: str = None

def extract_bpmn_elements(text: str) -> BPMNProcess:

    llm = ChatOpenAI(temperature=0.2, model="gpt-4")
    
    structured_llm = llm.with_structured_output(BPMNProcess)

    # Create the initial prompt template
    initial_prompt = """You are a BPMN (Business Process Model and Notation) expert. Analyze the given text and extract all BPMN elements into a structured format.

    Expected Output Structure:
    - process_name: The overall name of the business process
    - pools: Represent different organizations or major participants (e.g., "Company", "Customer")
        - id: Unique identifier (e.g., "pool_1")
        - name: Descriptive name of the pool
        - lanes: List of lanes within this pool
    - lanes: Represent departments or roles within pools
        - id: Unique identifier (e.g., "lane_1")
        - name: Name of department or role (e.g., "Customer Service Department")
        - parent_pool_id: ID of the containing pool
    - tasks: Represent activities or work being performed
        - id: Unique identifier (e.g., "task_1")
        - name: Description of the activity
        - type: One of [user, service, manual, script, business_rule]
        - lane_id: ID of the lane where this task belongs
        - incoming/outgoing: Lists of connection IDs
    - events: Represent start, intermediate, or end points
        - id: Unique identifier (e.g., "event_1")
        - name: Description of the event
        - type: One of [start, intermediate, end]
        - lane_id: ID of the lane where this event belongs
        - incoming/outgoing: Lists of connection IDs
    - gateways: Represent decision points or flow splits/joins
        - id: Unique identifier (e.g., "gateway_1")
        - name: Description of the decision point
        - type: One of [exclusive, parallel, inclusive, event_based]
        - conditions: Dictionary mapping outgoing connection IDs to their conditions
        - lane_id: ID of the lane where this gateway belongs
        - incoming/outgoing: Lists of connection IDs
    - connections: Represent flow between elements
        - source_id: ID of the source element
        - target_id: ID of the target element
        - condition: Optional condition for gateway paths

    Rules:
    1. Generate unique IDs for each element (e.g., "task_1", "gateway_2", etc.)
    2. Identify organizational units as pools or lanes
    3. Classify tasks by their appropriate type:
       - user: Human-performed tasks requiring user interaction
       - service: Automated tasks performed by a system
       - manual: Physical tasks performed without system interaction
       - script: Automated tasks using scripts
       - business_rule: Tasks governed by business rules
    4. Identify decision points as gateways and specify their type:
       - exclusive: Only one path can be taken
       - parallel: All paths are taken
       - inclusive: One or more paths can be taken
       - event_based: Path chosen based on occurring events
    5. Create connections between elements following the process flow
    6. Include conditions for exclusive gateways
    7. Ensure start and end events are present

    Text to analyze:
    {text}
    """
    
    # Get initial response
    final_prompt = initial_prompt.format(text=text)
    initial_response = structured_llm.invoke(final_prompt)

    # Create validation prompt
    validation_prompt = """You are a BPMN expert validator. Review the following BPMN process extraction and verify its correctness.
    If you find any issues, provide a corrected version with an explanation in the comment field.

    Validation Checklist:
    1. Process Flow Completeness:
       - Are all steps from the text represented?
       - Is the flow logical and continuous?
       - Are start and end events properly connected?

    2. Gateway Usage:
       - Are decisions properly modeled with appropriate gateway types?
       - Do exclusive gateways have clear conditions?
       - Are parallel activities properly modeled with parallel gateways?

    3. Task Classification:
       - Are tasks properly categorized (user, service, manual, etc.)?
       - Are they assigned to appropriate lanes?
       - Do they accurately represent the described work?

    4. Organizational Structure:
       - Are all participants represented in pools/lanes?
       - Is the hierarchy of pools and lanes correct?
       - Are elements properly assigned to lanes?

    5. Connection Validity:
       - Are all elements properly connected?
       - Are gateway splits and joins balanced?
       - Are conditions clear and complete?

    Original Text:
    {text}

    Generated BPMN Process:
    {initial_response}

    If you find any issues, output a corrected version with explanatory comments. If the original is correct, return it unchanged with a comment confirming its validity.
    """

    # Get validation response
    validation_prompt_filled = validation_prompt.format(
        text=text,
        initial_response=initial_response.model_dump_json()
    )
    
    structured_llm = llm.with_structured_output(BPMNProcessCorrected)

    validated_response = structured_llm.invoke(validation_prompt_filled)
    return validated_response

def convert_to_bpmn_xml(bpmn_process: BPMNProcess) -> str:
    # Create the root element with correct namespaces
    root = ET.Element('bpmn:definitions', {
        'xmlns:bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL',
        'xmlns:bpmndi': 'http://www.omg.org/spec/BPMN/20100524/DI',
        'xmlns:dc': 'http://www.omg.org/spec/DD/20100524/DC',
        'xmlns:di': 'http://www.omg.org/spec/DD/20100524/DI',
        'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
        'id': 'Definitions_1',
        'targetNamespace': 'http://bpmn.io/schema/bpmn'
    })

    # Create process with sanitized ID
    process_id = sanitize_id(bpmn_process.process_name)
    process = ET.SubElement(root, 'bpmn:process', {
        'id': f'Process_{process_id}',
        'isExecutable': 'false'
    })

    # Add all BPMN elements to the process
    for task in bpmn_process.tasks:
        task_id = sanitize_id(task.id)
        task_elem = ET.SubElement(process, 'bpmn:task', {
            'id': task_id,
            'name': task.name
        })

    for event in bpmn_process.events:
        event_id = sanitize_id(event.id)
        event_elem = ET.SubElement(process, 'bpmn:startEvent' if event.type == 'start' else 'bpmn:endEvent', {
            'id': event_id,
            'name': event.name
        })

    # Add diagram information
    diagram = ET.SubElement(root, 'bpmndi:BPMNDiagram', {
        'id': 'BPMNDiagram_1'
    })
    
    plane = ET.SubElement(diagram, 'bpmndi:BPMNPlane', {
        'id': 'BPMNPlane_1',
        'bpmnElement': f'Process_{process_id}'
    })

    # Return the formatted XML
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def sanitize_id(text: str) -> str:
    """Sanitize text to be used as XML ID"""
    # Remove special characters and spaces
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', text)
    # Ensure it starts with a letter or underscore
    if sanitized[0].isdigit():
        sanitized = f'_{sanitized}'
    return sanitized

def save_bpmn_diagram(bpmn_process: BPMNProcess, filename: str):
    """Save the BPMN process as a BPMN 2.0 XML file"""
    xml_string = convert_to_bpmn_xml(bpmn_process)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(xml_string)

txt = """

Le processus de souscription à une assurance débute généralement lorsque
le client remplit un formulaire de demande, incluant ses informations personnelles et
les détails précis du bien ou de la personne à assurer. Parfois, les clients préfèrent
envoyer ces informations par courrier, bien que ce ne soit pas recommandé. Une fois le
formulaire reçu, le Service Clientèle s'assure que toutes les données nécessaires sont
présentes. Si des informations sont manquantes, l'agent du Service Clientèle peut
devoir contacter le client pour compléter le dossier, sauf s'il y a une surcharge de travail
ce jour-là.
Après cette étape, le dossier est transmis au Département de Souscription, où
les analystes de risques évaluent les risques potentiels. Cette évaluation peut inclure
une vérification des antécédents du client, une analyse du marché actuel de
l'assurance, et parfois même une inspection physique du bien à assurer réalisée par
un inspecteur. Il est intéressant de noter que des facteurs externes, comme les
conditions météorologiques ou les tendances économiques, peuvent influencer cette
évaluation. Pendant ce temps, le client pourrait recevoir des offres promotionnelles
pour d'autres produits envoyées par le Service Marketing, mais cela n'affecte pas le
processus de souscription en lui-même.
Si l'évaluation des risques est favorable, la souscription est approuvée. Sinon, elle peut
être rejetée ou nécessiter des ajustements, tels que l'augmentation de la prime ou
l'ajout de clauses spécifiques. Une fois approuvée, la police d'assurance est émise par
le Service Émission des Polices. Le client reçoit alors sa police par courrier
électronique ou postal, selon sa préférence, bien que des retards puissent survenir en
cas de jours fériés.
Il est crucial que le client paie la prime pour que la police soit activée. Certains clients
oublient cette étape, ce qui peut entraîner des complications en cas de sinistre. De
plus, si le client a des questions ou des préoccupations, il peut contacter le Service
Clientèle, bien que les temps d'attente puissent varier en fonction de l'heure et du jour.
La satisfaction du client est notre priorité absolue, même si nous ne pouvons pas
toujours répondre immédiatement à toutes les demandes. Enfin, il convient de
mentionner que le processus peut être accéléré si le client utilise notre application
mobile, qui est disponible sur iOS et Android, à moins qu'il ne rencontre des problèmes
techniques, auquel cas il peut contacter le Service Support Technique."""

validated_process = extract_bpmn_elements(txt)
save_bpmn_diagram(validated_process, 'insurance_process.bpmn')

