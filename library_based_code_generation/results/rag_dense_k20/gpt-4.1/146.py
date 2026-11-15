```python
from kubernetes import client, config

DEPLOYMENT_NAME = "example-deployment"
SERVICE_NAME = "example-service"
INGRESS_NAME = "example-ingress"
NAMESPACE = "default"
CONTAINER_IMAGE = "nginx:1.25"
CONTAINER_PORT = 80
SERVICE_PORT = 80
INGRESS_HOST = "example.local"
INGRESS_PATH = "/"

def create_deployment(api_instance):
    container = client.V1Container(
        name="example-container",
        image=CONTAINER_IMAGE,
        ports=[client.V1ContainerPort(container_port=CONTAINER_PORT)]
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": DEPLOYMENT_NAME}),
        spec=client.V1PodSpec(containers=[container])
    )
    spec = client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(match_labels={"app": DEPLOYMENT_NAME}),
        template=template
    )
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=DEPLOYMENT_NAME),
        spec=spec
    )
    api_instance.create_namespaced_deployment(
        namespace=NAMESPACE,
        body=deployment
    )

def create_service(api_instance):
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=SERVICE_NAME),
        spec=client.V1ServiceSpec(
            selector={"app": DEPLOYMENT_NAME},
            ports=[client.V1ServicePort(
                port=SERVICE_PORT,
                target_port=CONTAINER_PORT
            )],
            type="ClusterIP"
        )
    )
    api_instance.create_namespaced_service(
        namespace=NAMESPACE,
        body=service
    )

def create_ingress(api_instance):
    ingress_backend = client.V1IngressBackend(
        service=client.V1IngressServiceBackend(
            name=SERVICE_NAME,
            port=client.V1ServiceBackendPort(number=SERVICE_PORT)
        )
    )
    ingress_rule = client.V1IngressRule(
        host=INGRESS_HOST,
        http=client.V1HTTPIngressRuleValue(
            paths=[
                client.V1HTTPIngressPath(
                    path=INGRESS_PATH,
                    path_type="Prefix",
                    backend=ingress_backend
                )
            ]
        )
    )
    ingress_spec = client.V1IngressSpec(
        rules=[ingress_rule]
    )
    ingress = client.V1Ingress(
        api_version="networking.k8s.io/v1",
        kind="Ingress",
        metadata=client.V1ObjectMeta(name=INGRESS_NAME),
        spec=ingress_spec
    )
    api_instance.create_namespaced_ingress(
        namespace=NAMESPACE,
        body=ingress
    )

def main():
    config.load_kube_config()
    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    networking_v1 = client.NetworkingV1Api()

    create_deployment(apps_v1)
    create_service(core_v1)
    create_ingress(networking_v1)

if __name__ == "__main__":
    main()
```