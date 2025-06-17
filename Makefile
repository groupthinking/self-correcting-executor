up:
	docker-compose up -d

down:
	docker-compose down

build:
	docker-compose build

logs:
	docker-compose logs -f --tail=100

health:
	curl -f http://localhost:8080/health || echo "Health check failed" 