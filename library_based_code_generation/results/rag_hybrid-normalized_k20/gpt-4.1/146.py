```python
from kubernetes import client, config

def create_deployment(api_instance, namespace, name, image, container_port):
    container = client.V1Container(
        name=name,
        image=image,
        ports=[client.V1ContainerPort(container_port=container_port)]
    )
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": name}),
        spec=client.V1PodSpec(containers=[container])
    )
    spec = client.V1DeploymentSpec(
        replicas=1,
        selector=client.V1LabelSelector(match_labels={"app": name}),
        template=template
    )
    deployment = client.V1Deployment(
        api_version="apps/v1",
        kind="Deployment",
        metadata=client.V1ObjectMeta(name=name),
        spec=spec
    )
    api_instance.create_namespaced_deployment(
        namespace=namespace,
        body=deployment
    )

def create_service(api_instance, namespace, name, port):
    service = client.V1Service(
        api_version="v1",
        kind="Service",
        metadata=client.V1ObjectMeta(name=name),
        spec=client.V1ServiceSpec(
            selector={"app": name},
            ports=[client.V1ServicePort(port=port, target_port=port)],
            type="ClusterIP"
        )
    )
    api_instance.create_namespaced_service(
        namespace=namespace,
        body=service
    )

def create_ingress(api_instance, namespace, name, service_name, service_port, host, path):
    ingress_backend = client.V1IngressBackend(
        service=client.V1IngressServiceBackend(
            name=service_name,
            port=client.V1ServiceBackendPort(number=service_port)
        )
    )
    ingress_rule = client.V1IngressRule(
        host=host,
        http=client.V1HTTPIngressRuleValue(
            paths=[
                client.V1HTTPIngressPath(
                    path=path,
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
        metadata=client.V1ObjectMeta(name=name),
        spec=ingress_spec
    )
    api_instance.create_namespaced_ingress(
        namespace=namespace,
        body=ingress
    )

def main():
    config.load_kube_config()
    namespace = "default"
    deployment_name = "my-app"
    image = "nginx:1.25"
    container_port = 80
    service_name = deployment_name
    service_port = 80
    ingress_name = "my-app-ingress"
    host = "myapp.example.com"
    path = "/"

    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()
    networking_v1 = client.NetworkingV1Api()

    create_deployment(apps_v1, namespace, deployment_name, image, container_port)
    create_service(core_v1, namespace, service_name, service_port)
    create_ingress(networking_v1, namespace, ingress_name, service_name, service_port, host, path)

if __name__ == "__main__":
    main()
```